import os
import threading
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, TextIteratorStreamer
import torch
import gradio as gr

# from transformers.generation.streamers import TextIteratorStreamer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Sign in to HuggingFace Hub
# hf_token = os.getenv('HF_TOKEN')
# login(hf_token, add_to_git_credential=True)

# Sign in to OpenAI using Secrets in Colab
openai_api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)

# Use the Whisper OpenAI model to convert the Audio to Text
# If you'd prefer to use an Open Source model, class student Youssef has contributed an open source version
# which I've added to the bottom of this colab
audio_filename = "denver_extract.mp3"
audio_file = open(audio_filename, "rb")
transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
print(transcription)

system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

response = tokenizer.decode(outputs[0])

display(Markdown(response))

def audio2text2(audio_file: str, system_message: str, user_prompt: str, device: str='cpu'):
    AUDIO_MODEL = "whisper-1"
    LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Sign in to HuggingFace Hub
    hf_token = os.getenv('HF_TOKEN')
    login(hf_token, add_to_git_credential=True)

    # Sign in to OpenAI using Secrets in Colab
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai = OpenAI(api_key=openai_api_key)

    audio = open(audio_file, "rb")
    transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio, response_format="text")

    user_prompt = user_prompt.replace("{{transcription}}", transcription)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

    response = tokenizer.decode(outputs[0])
    # display(Markdown(response))

    return response

def audio2text(audio_file: str, device: str='cpu', model: str='openai/whisper-medium'):
    AUDIO_MODEL = "openai/whisper-medium" if model == None or model == '' else model
    speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        AUDIO_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    # speech_model.to(device)
    processor = AutoProcessor.from_pretrained(AUDIO_MODEL)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=speech_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
    )

    # Use the Whisper OpenAI model to convert the Audio to Text
    result = pipe(audio_file, return_timestamps=True)
    transcription = result["text"]
    print(result)
    return transcription


# Load model and tokenizer
def load_model(model_name):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config)
    return tokenizer, model

# define different generating functions:
#   1- full response
#   2- low level streaming response
#   3- low level streaming response

def generate_full(tokenizer: AutoTokenizer,
                  model: AutoModelForCausalLM,
                  user_input: str,
                  max_tokens: int=2000,
                  device: str='cpu'):
    global messages
    # Append the user's new message to the conversation history
    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)
    outputs = model.generate(inputs, max_new_tokens=max_tokens)
    response = tokenizer.decode(outputs[0])
    return response

def generate_stream_low_level(tokenizer, model, user_input, max_tokens=2000):
    global messages
    # Append the user's new message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Prepare the initial input
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    # Generate up to 2000 tokens
    for _ in range(max_tokens):
        outputs = model(input_ids)  # Get the model's output (logits) for the given input IDs
        # Select the token with the highest probability from the last position's logits
        next_token_id = outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)  # Append new token
        next_token = tokenizer.decode(next_token_id[0])  # Decode and print
        # flush=True ensures the output is immediately written to the console.
        # By default, print output is buffered, so it may not appear instantly.
        # flush=True forces the buffer to flush, making real-time output possible.
        print(next_token, end="", flush=True)

        if next_token_id.item() == tokenizer.eos_token_id:  # Stop if EOS token
            break

def generate_stream_high_level(tokenizer, model, user_input, max_tokens=2000, device='cpu'):
    global messages
    # Append the user's new message to the conversation history
    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)
    # we skip using TextStreamer() here cause it streams back results to stdout and thats not what we want in gradio app
    # and we use TextIteratorStreamer() instead

    # Initialize the TextIteratorStreamer for streaming output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        decode_kwargs={"skip_special_tokens": True}
    )

    # Run the generation process in a separate thread
    thread = threading.Thread(
        target=model.generate,
        kwargs={"inputs": inputs, "max_new_tokens": max_tokens, "streamer": streamer}
    )
    thread.start()

    # Stream and print the output progressively
    for text_chunk in streamer:
        filtered_chunk = text_chunk.replace("<|eot_id|>", "")  # Remove special tokens if present
        print(filtered_chunk, end="")  # Print without adding new lines


# Load the model and tokenizer
tokenizer, model = load_model(LLAMA)

# initialize the messages history, the max tokens for the model, and the user_input
messages = [
    {"role": "system", "content": "You are a helpful assistant"}
]

max_tokens = 2000

user_input = "What is the meaning of life? Answer in markdown and in 5 lines maximum."

generate_full(tokenizer, model, user_input)

generate_stream_high_level(tokenizer, model, user_input)


########################## Gradio ##########################
# define the streaming function for gradio (using yield)
def generate_stream(user_input, device='cpu'):
    # Global variables for modifications
    global tokenizer, model, messages, max_tokens

    # Step 1: Append the user's new message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Step 2: Tokenize the input messages and convert them into a tensor
    # - apply_chat_template: Formats the messages according to the model's expected input format.
    # - return_tensors="pt": Returns the result as a PyTorch tensor.
    # - add_generation_prompt=True: Adds a special prompt or token for generation.
    # - .to("cuda"): Moves the tensor to the GPU for faster computation.
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)
    # Type: torch.Tensor of shape [batch_size, sequence_length].to("cuda")

    # Initialize an empty string to accumulate the generated result
    result = ""

    # Step 3: Start generating tokens in a loop, up to a maximum of 2000 tokens
    for _ in range(max_tokens):
        # Step 4: Pass the current input sequence to the model to predict the next token
        # - outputs.logits: Contains the raw prediction scores (logits) for all possible tokens.
        # - Shape of outputs.logits: [batch_size, sequence_length, vocab_size].
        # - outputs.logits[:, -1]: Selects the logits for the last token position (shape: [batch_size, vocab_size]).
        outputs = model(input_ids)
        # Type: transformers.modeling_outputs.CausalLMOutputWithPast containing logits of shape [batch_size, sequence_length, vocab_size]

        # Step 5: Find the token ID with the highest score (greedy decoding)
        # - argmax(dim=-1): Selects the index of the maximum value along the vocab_size dimension.
        # - unsqueeze(-1): Adds a new dimension at the last position, resulting in a shape of [batch_size, 1].
        next_token_id = outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
        # Type: torch.Tensor of shape [batch_size, 1].unsqueeze(-1)

        # Step 6: Append the newly generated token ID to the input_ids tensor
        # - torch.cat(): Concatenates the current input_ids with the next_token_id along the last dimension.
        # - This updates input_ids to include the newly generated token, so the model can use the updated sequence in the next iteration.
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        # Type: torch.Tensor of shape [batch_size, updated_sequence_length]

        # Step 7: Decode the newly generated token ID into a human-readable string
        # - tokenizer.decode(): Converts the token ID into its corresponding string.
        # - skip_special_tokens=True: Ensures special tokens like <eos> (end-of-sequence) are not included in the output.
        next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        # Type: str representing the decoded token

        # Step 8: Accumulate the decoded token into the result string
        result += next_token

        # Step 9: Yield the accumulated result for streaming output
        # - yield allows the function to return partial results without stopping, enabling real-time streaming.
        yield result

        # Step 10: Check if the model predicted the end-of-sequence (EOS) token
        # - tokenizer.eos_token_id: The special token ID representing EOS.
        # - If EOS is detected, break the loop to stop further generation.
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Append the final assistant response to the conversation history
    messages.append({"role": "assistant", "content": result})


# optimize the streaming function for gradio (using TextIteratorStreamer)
def generate_stream_optimized(user_input, device='cpu'):
    # Global variables for modifications
    global tokenizer, model, messages, max_tokens

    # Step 1: Append the user's new message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Step 2: Prepare the inputs for the model by applying the chat template
    # The inputs include the conversation history and the user's latest message
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)
    # we skip using TextStreamer() here cause it streams back results to stdout and thats not what we want in gradio app
    # we use TextIteratorStreamer() instead

    # Step 3: Initialize the TextIteratorStreamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,  # Ensures that the input prompt is not repeatedly included in the streamed output.
        decode_kwargs={"skip_special_tokens": True}  # Filters out special tokens (e.g., <s>, </s>, <pad>, <cls>, <sep>) from the generated text.
    )

    # Step 4: Create a thread to run the generation process in the background
    thread = threading.Thread(
        target=model.generate,  # Specifies that the model's `generate` method will be run in the thread.
        kwargs={                           # Passes the arguments required for text generation
            "inputs": inputs,              # The tokenized input prompt for the model.
            "max_new_tokens": max_tokens,  # Limits the number of tokens to be generated.
            "streamer": streamer           # The TextIteratorStreamer to handle streaming the output.
        }
    )

    # Step 5: Start the thread to begin the generation process
    thread.start()

    # Step 6: Initialize an empty string to accumulate the growing output
    accumulated_reply = ""

    # Step 7: Stream the output progressively
    for text_chunk in streamer:  # Iterate over each chunk of text streamed by the model
        # Filter out any unexpected special tokens manually if they appear to ensure a clean output
        # `<|eot_id|>` is a special token (e.g., end-of-text marker) that may still appear in some outputs
        filtered_chunk = text_chunk.replace("<|eot_id|>", "")

        # Append the filtered chunk to the accumulated text that holds all the generated text seen so far
        accumulated_reply += filtered_chunk

        # Yield the accumulated text to the calling function/UI for progressive updates,
        # ensuring the output is continuously refreshed with new content
        yield accumulated_reply

    # Step 8: Append the final assistant response to the conversation history
    messages.append({"role": "assistant", "content": accumulated_reply})

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chat with AI (Streaming Enabled)")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Your message", placeholder="Type something...")
            output_box = gr.Markdown(label="AI Response", min_height=50)
            send_button = gr.Button("Send")

    send_button.click(fn=generate_stream_optimized, inputs=user_input, outputs=output_box)

demo.launch()
####################### END - Gradio #######################