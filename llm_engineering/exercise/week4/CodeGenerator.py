import os
import io
import sys
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
import gc
import gradio as gr
import subprocess
import torch
from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

from Excecutor import Excecutor

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')


class CodeGenerator:

    hf_token: str = os.environ['HF_TOKEN']
    model: str = None
    system_message: str = ''

    OPENAI_MODEL = "gpt-4.1"
    CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
    code_qwen = "Qwen/CodeQwen1.5-7B-Chat"
    code_gemma = "google/codegemma-7b-it"
    CODE_QWEN_URL = "https://h1vdol7jxhje3mpn.us-east-1.aws.endpoints.huggingface.cloud"
    CODE_GEMMA_URL = "https://c5hggiyqachmgnqg.us-east-1.aws.endpoints.huggingface.cloud"

    openai: OpenAI = None
    claude: anthropic.Anthropic = None

    # Quantization Config - this allows us to load the model into memory and use less memory (need CUDA)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    def __init__(self, model: str=None, hf_token: str=None):
        self.hf_token = hf_token if hf_token is not None and hf_token != '' else os.environ['HF_TOKEN']
        login(self.hf_token, add_to_git_credential=True)

        # self.model = model if model  is not None and model != '' else self.code_qwen

        self.system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
        self.system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
        self.system_message += "The C++ response needs to produce an identical output in the fastest possible time."

        self.openai = OpenAI()
        self.claude = anthropic.Anthropic()

    def has_cuda(self):
        return torch.cuda.is_available()

    def get_device(self):
        return 'cuda' if self.has_cuda() else 'cpu'

    def user_prompt_for(self, python_codes: str=''):
        user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
        user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
        user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
        user_prompt += python_codes
        return user_prompt

    def messages_for(self, python_codes: str=''):
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_prompt_for(python_codes)}
        ]

    def write_output(self, cpp: str='', filename: str='optimized_test.cpp'):
        code = cpp.replace("```cpp","").replace("```","")
        with open(f"{filename}", "w") as f:
            f.write(code)

    def optimize_gpt(self, python_codes, filename: str='optimized_gpt.cpp'):
        stream = self.openai.chat.completions.create(model=self.OPENAI_MODEL, messages=self.messages_for(python_codes), stream=True)
        reply = ""
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            reply += fragment
            print(fragment, end='', flush=True)
        self.write_output(reply, filename=filename)

    def optimize_claude(self, python_codes: str='', filename: str='optimized_claude.cpp'):
        result = self.claude.messages.stream(
            model=self.CLAUDE_MODEL,
            max_tokens=2000,
            system=self.system_message,
            messages=[{"role": "user", "content": self.user_prompt_for(python_codes)}],
        )
        reply = ""
        with result as stream:
            for text in stream.text_stream:
                reply += text
                print(text, end="", flush=True)
        self.write_output(reply, filename=filename)

    def stream_gpt(self, python_codes: str=''):
        stream = self.openai.chat.completions.create(model=self.OPENAI_MODEL, messages=self.messages_for(python_codes), stream=True)
        reply = ""
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            reply += fragment
            yield reply.replace('```cpp\n','').replace('```','')

    def stream_claude(self, python_codes: str=''):
        result = self.claude.messages.stream(
            model=self.CLAUDE_MODEL,
            max_tokens=2000,
            system=self.system_message,
            messages=[{"role": "user", "content": self.user_prompt_for(python_codes)}],
        )
        reply = ""
        with result as stream:
            for text in stream.text_stream:
                reply += text
                yield reply.replace('```cpp\n','').replace('```','')

    def stream_code_qwen(self, python_codes: str='', model: str=None, clear_cache: bool=True):
        model = model if model is not None and model != '' else self.code_qwen
        device = self.get_device()
        max_new_tokens = 500 # 3000
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        text = tokenizer.apply_chat_template(self.messages_for(python_codes),
                                            #  tokenize=False,
                                             return_tensors="pt",
                                             add_generation_prompt=True
                                            ).to(device)
        # Way1: using http client
        # client = InferenceClient(self.CODE_QWEN_URL, token=self.hf_token)
        # stream = client.text_generation(text, stream=True, details=True, max_new_tokens=max_new_tokens)
        # result = ""
        # for r in stream:
        #     result += r.token.text
        #     yield result

        quantization_config = self.quant_config if self.has_cuda() else None
        _model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            quantization_config=quantization_config
        )

        # Way2: using model
        # Stream
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        outputs = _model.generate(
            text,
            streamer=streamer,
            max_new_tokens=max_new_tokens
        )

        result = ""
        for new_text in streamer:
            print(f'new_text: {new_text}')
            result += new_text
            yield result

        # Way3: thread
        # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        # generation_kwargs = dict(text, streamer=streamer, max_new_tokens=max_new_tokens)
        # thread = Thread(target=model.generate, kwargs=generation_kwargs)
        # thread.start()
        # result = ""
        # for new_text in streamer:
        #     result += new_text
        #     yield result

        # Clear cache
        if clear_cache:
            del model, text, tokenizer, outputs, streamer
            gc.collect()
            torch.cuda.empty_cache()


    def optimize(self, python_codes: str='', model: str=''):
        if model == "GPT":
            result = self.stream_gpt(python_codes)
        elif model == "Claude":
            result = self.stream_claude(python_codes)
        elif model == "CodeQwen":
            result = self.stream_code_qwen(python_codes)
        else:
            raise ValueError(f"Unknown model: {model}")
        for stream_so_far in result:
            yield stream_so_far

    def execute_python(self, code):
        try:
            print('Executing the python codes...')
            output = io.StringIO()
            sys.stdout = output
            exec(code)
            print('Executed the python codes...')
        finally:
            sys.stdout = sys.__stdout__
        return output.getvalue()

    def execute_cpp(self, code):
        print('Executing the C++ codes...')
        filename = 'optimized.cpp'
        outfile = "optimized.exe"
        print(f'Saving the {filename} file for the C++ codes...')
        self.write_output(code, filename=filename)
        print(f'Saved the {filename} file')
        try:
            # compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", "optimized", "optimized.cpp"]
            print(f'Building ({filename} file...')
            compile_cmd = ["g++", "-g", filename, "-o", outfile]
            print(f'Built the `{filename}` file into `{outfile}` file...')
            compile_result = subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
            print(f'Running the `{outfile}` file...')
            run_cmd = [f"./{outfile}"]
            run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
            print(f'Runned the ({outfile} file...')
            return run_result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred:\n{e.stderr}"

code_generator = CodeGenerator()
compiler_cmd = Excecutor.compile_c("optimized")

pi = """
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

python_hard = """# Be careful to support large number sizes

def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        yield value

def max_subarray_sum(n, seed, min_val, max_val):
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum

def total_max_subarray_sum(n, initial_seed, min_val, max_val):
    total_sum = 0
    lcg_gen = lcg(initial_seed)
    for _ in range(20):
        seed = next(lcg_gen)
        total_sum += max_subarray_sum(n, seed, min_val, max_val)
    return total_sum

# Parameters
n = 10000         # Number of random numbers
initial_seed = 42 # Initial seed for the LCG
min_val = -10     # Minimum value of random numbers
max_val = 10      # Maximum value of random numbers

# Timing the function
import time
start_time = time.time()
result = total_max_subarray_sum(n, initial_seed, min_val, max_val)
end_time = time.time()

print("Total Maximum Subarray Sum (20 runs):", result)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
"""

def select_sample_program(sample_program):
    if sample_program == "pi":
        return pi
    if sample_program == "python_hard":
        return python_hard
    return "Type your Python program here"



css = """
.python {background-color: #000000;}
.cpp {background-color: #000fff;}
"""
# css = """
# .python {background-color: #306998;}
# .cpp {background-color: #050;}
# """

# with gr.Blocks(css=css) as ui:
#     gr.Markdown("## Convert code from Python to C++")
#     with gr.Row():
#         python = gr.Textbox(label="Python code:", value=python_hard, lines=10)
#         cpp = gr.Textbox(label="C++ code:", lines=10)
#     with gr.Row():
#         sample_program = gr.Radio(["pi", "python_hard"], label="Sample program", value="python_hard")
#         model = gr.Dropdown(["GPT", "Claude",  "CodeQwen"], label="Select model", value="GPT")
#     with gr.Row():
#         convert = gr.Button("Convert code")
#     with gr.Row():
#         python_run = gr.Button("Run Python")
#         cpp_run = gr.Button("Run C++")
#     with gr.Row():
#         python_out = gr.TextArea(label="Python result:", elem_classes=["python"])
#         cpp_out = gr.TextArea(label="C++ result:", elem_classes=["cpp"])

#     convert.click(code_generator.optimize, inputs=[python, model], outputs=[cpp])
#     python_run.click(code_generator.execute_python, inputs=[python], outputs=[python_out])
#     cpp_run.click(code_generator.execute_cpp, inputs=[cpp], outputs=[cpp_out])

# ui.launch(inbrowser=True)

with gr.Blocks(css=css) as ui:
    gr.Markdown("## Convert code from Python to C++")
    with gr.Row():
        python = gr.Textbox(label="Python code:", value=python_hard, lines=10)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        with gr.Column():
            sample_program = gr.Radio(["pi", "python_hard"], label="Sample program", value="python_hard")
            model = gr.Dropdown(["GPT", "Claude", "CodeQwen"], label="Select model", value="GPT")
        with gr.Column():
            architecture = gr.Radio([compiler_cmd[0]], label="Architecture", interactive=False, value=compiler_cmd[0])
            compiler = gr.Radio([compiler_cmd[1]], label="Compiler", interactive=False, value=compiler_cmd[1])
    with gr.Row():
        convert = gr.Button("Convert code")
    with gr.Row():
        python_run = gr.Button("Run Python")
        if not compiler_cmd[1] == "Unavailable":
            cpp_run = gr.Button("Run C++")
        else:
            cpp_run = gr.Button("No compiler to run C++", interactive=False)
    with gr.Row():
        python_out = gr.TextArea(label="Python result:", elem_classes=["python"])
        cpp_out = gr.TextArea(label="C++ result:", elem_classes=["cpp"])

    sample_program.change(select_sample_program, inputs=[sample_program], outputs=[python])
    convert.click(code_generator.optimize, inputs=[python, model], outputs=[cpp])
    python_run.click(code_generator.execute_python, inputs=[python], outputs=[python_out])
    cpp_run.click(code_generator.execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)
