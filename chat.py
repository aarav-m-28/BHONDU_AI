import torch
from model import ModelArgs, Transformer
from transformers import AutoTokenizer
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "llama3_custom.pt"
tokenizer_path = "llama-3-3b"
print(device)
print(f"Loading tokenizer from {tokenizer_path}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

print("Initializing model...")
# We need to manually set args to match 3B
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

# Set default dtype to bfloat16 to save memory (Critical for 8GB VRAM)
torch.set_default_dtype(torch.bfloat16)
model = Transformer(args)
torch.set_default_dtype(torch.float32) # Reset to default

print(f"Loading weights from {model_path}...")
if os.path.exists(model_path):
    # Load directly to device if possible, or CPU then move
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict # Free CPU memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Error: Model weights not found. Run convert_hf_to_custom.py first.")
    exit()

print("Moving model to GPU...")
model.to(device)
torch.cuda.empty_cache()
model.eval()

print("==================================================")
print("Chat with Bhondu")
print("==================================================")

messages = [
    {"role": "system", "content": "You are Bhondu, a helpful and smart AI assistant."}
]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit':
        break
        
    messages.append({"role": "user", "content": user_input})
    
    # Format prompt using tokenizer template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    print("AI: ", end='', flush=True)
    with torch.no_grad():
        # 1. Prefill (Process the prompt)
        logits = model(tokens, start_pos=0)
        
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        text = tokenizer.decode(next_token[0])
        print(text, end='', flush=True)
        
        # 2. Generation Loop (Incremental)
        cur_pos = tokens.shape[1] # Current position after prompt
        tokens = torch.cat((tokens, next_token), dim=1)
        
        full_response = text
        
        for i in range(200):
            # print(f"[DEBUG] Step {i} start. cur_pos={cur_pos}")
            # Pass only the LAST token, but with correct start_pos
            # The model will use cached KV for previous tokens
            logits = model(next_token, start_pos=cur_pos)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat((tokens, next_token), dim=1)
            cur_pos += 1
            
            # Decode just the new token
            text = tokenizer.decode(next_token[0])
            print(text, end='', flush=True)
            full_response += text
            
            if next_token.item() == tokenizer.eos_token_id or "user" in text.lower(): # Stop conditions
                break
    
    messages.append({"role": "assistant", "content": full_response})
