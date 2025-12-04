import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 3072 # 3B model dimension
    n_layers: int = 28
    n_heads: int = 24
    n_kv_heads: int = 8 # Grouped Query Attention
    vocab_size: int = 128256 # Llama 3 vocab
    multiple_of: int = 1024 # SwiGLU hidden layer size multiple
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_batch_size: int = 8
    max_seq_len: int = 2048 # Can be extended to 128k

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) or freqs_cis.shape == (x.shape[1], x.shape[-1] // 2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    if freqs_cis.shape[-1] == x.shape[-1] // 2:
        shape[-1] = x.shape[-1] // 2
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # Llama 3 uses "rotate half" logic, but let's check if we can adapt our complex logic.
    # Actually, standard Llama implementation in PyTorch often uses complex numbers too IF the weights are permuted.
    # But since we loaded HF weights directly without permutation, we must match HF's RoPE.
    # HF RoPE: (x * cos) + (rotate_half(x) * sin)
    # This implies x is NOT treated as pairs (0,1), but as (0, dim/2).
    
    # We need to compute cos/sin from freqs_cis
    freqs_cis = reshape_for_broadcast(freqs_cis, xq)
    
    # We need to split xq into real/imag parts assuming the "rotate half" structure?
    # No, complex numbers naturally assume pairs (real, imag).
    # If HF weights are trained with rotate_half, then x[0] and x[dim/2] are the pair.
    # My current code assumes x[0] and x[1] are the pair.
    
    # To fix this without complex math rewrite:
    # We can permute xq to bring pairs together:
    # xq: [..., dim] -> reshape [..., 2, dim/2] -> transpose -> [..., dim/2, 2] -> view_as_complex
    
    # Wait, simpler: Implement explicit rotate_half logic.
    
    xq_r, xq_i = freqs_cis.real, freqs_cis.imag
    xk_r, xk_i = freqs_cis.real, freqs_cis.imag
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # We need to repeat freqs to match full dim
    # freqs_cis is [seq, dim/2] (complex)
    # We need cos, sin as [seq, dim]
    
    cos = freqs_cis.real
    sin = freqs_cis.imag
    
    # Repeat for the two halves
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    
    # Apply
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV Cache
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV Cache Logic
        if self.cache_k is None or self.cache_k.shape[1] < start_pos + seqlen:
            # Initialize or resize cache (simple dynamic resizing)
            # In production, we'd pre-allocate max_seq_len
            pass # We rely on the caller to handle state or we implement simple caching here
        
        # For simplicity in this custom implementation, we will assume the caller passes the full sequence 
        # OR we implement a proper cache. 
        # Let's implement a proper cache stored in the module for inference.
        
        if self.training:
            # No caching during training
            keys, values = xk, xv
        else:
            # Inference caching
            if self.cache_k is None:
                self.cache_k = torch.zeros((bsz, 2048, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros((bsz, 2048, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)
            
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

        # Repeat KV heads for GQA
        keys = torch.repeat_interleave(keys, self.n_rep, dim=2)
        values = torch.repeat_interleave(values, self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: int = 0):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.params = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2, self.params.rope_theta
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask, start_pos)
        h = self.norm(h)
        output = self.output(h).float()
        return output
