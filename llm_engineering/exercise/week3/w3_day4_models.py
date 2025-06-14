import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, TextIteratorStreamer
import torch
import gc

load_dotenv(override=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct" # exercise for you
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # If this doesn't fit it your GPU memory, try others from the hub

# Quantization Config - this allows us to load the model into memory and use less memory (need CUDA)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

def generate(model, messages: str, device: str='cpu', clear_cache: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False).to(device)
    # streamer = TextStreamer(tokenizer)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    quantization_config = quant_config if device == 'cuda' else None
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quantization_config)
    outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)

    result = ""
    for new_text in streamer:
        print(f'new text => {new_text}')
        result += new_text
    print(result)

    # Clear cache
    if clear_cache:
        del model, inputs, tokenizer, outputs, streamer
        gc.collect()
        torch.cuda.empty_cache()

python_codes = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""
system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
system_message += "The C++ response needs to produce an identical output in the fastest possible time."
user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
user_prompt += python_codes
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]
# Test model PHI3
# messages = [
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
# ]
PHI3 = 'Qwen/CodeQwen1.5-7B-Chat'
generate(PHI3, messages)

# Test model GEMMA2
# messages = [
#     {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
#   ]
# generate(GEMMA2, messages)