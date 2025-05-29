import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'


hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct" # exercise for you
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # If this doesn't fit it your GPU memory, try others from the hub

# Quantization Config - this allows us to load the model into memory and use less memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

def generate(model, messages, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False).to(device)
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quant_config)
    outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)
    # del model, inputs, tokenizer, outputs, streamer
    # gc.collect()
    # torch.cuda.empty_cache()
    print(outputs)

# Test model PHI3
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]
generate(PHI3, messages)

# Test model GEMMA2
# messages = [
#     {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
#   ]
# generate(GEMMA2, messages)