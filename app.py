import torch
from model import ModelArgs, Transformer
from transformers import AutoTokenizer
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio

app = FastAPI()

# --- Model Setup (Same as chat.py) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "llama3_custom.pt"
tokenizer_path = "llama-3-3b"

print(f"Loading tokenizer from {tokenizer_path}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

print("Initializing model...")
args = ModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    norm_eps=1e-5,
    rope_theta=500000.0,
    max_seq_len=2048
)

torch.set_default_dtype(torch.bfloat16)
model = Transformer(args)
torch.set_default_dtype(torch.float32)

print(f"Loading weights from {model_path}...")
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Error: Model weights not found.")
    exit()

print("Moving model to GPU...")
model.to(device)
torch.cuda.empty_cache()
model.eval()
print("Model Ready!")

# --- API ---

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message
    history = request.history

    # Construct messages
    messages = [
        {"role": "system", "content": "You are Bhondu, a helpful and smart AI assistant."}
    ]
    # Add history (limit to last 10 turns to save context)
    for msg in history[-10:]: 
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_input})

    # Format prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    async def generate_stream():
        with torch.no_grad():
            # Prefill
            logits = model(tokens, start_pos=0)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            text = tokenizer.decode(next_token[0])
            yield text
            
            cur_pos = tokens.shape[1]
            current_tokens = torch.cat((tokens, next_token), dim=1)
            
            for _ in range(200): # Max new tokens
                logits = model(next_token, start_pos=cur_pos)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                current_tokens = torch.cat((current_tokens, next_token), dim=1)
                cur_pos += 1
                
                text = tokenizer.decode(next_token[0])
                yield text
                
                if next_token.item() == tokenizer.eos_token_id or "user" in text.lower():
                    break
                
                # Small delay to allow event loop to run (optional, but good for async)
                await asyncio.sleep(0)

    return StreamingResponse(generate_stream(), media_type="text/plain")

# Serve Static Files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
