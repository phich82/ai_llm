import base64
from io import BytesIO
import json
import os
import sys
import time

from PIL import Image
from openai import OpenAI
from openai.types import ImagesResponse
from openai.types.chat import ChatCompletionMessageToolCall



# Current fdir
script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
# Append the relative paths
sys.path.append("..")

import gradio as gr
from matplotlib import lines

from LLM import LLM
from enums.ServiceEnum import ServiceEnum
from Crawler import Crawler
from OpenChat import OpenChat, ResponseChat
from ImageGenerator import ImageGenerator
from Audio import Audio


########################## TOOLS #########################
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

# Useful function
def get_ticket_price(destination_city: str=None):
    destination_city = '' if destination_city is None else destination_city.strip()
    if destination_city != '':
        print(f"Tool [get_ticket_price] called for {destination_city}")
    city = destination_city.lower()
    price = ticket_prices.get(city, "Unknown")
    if destination_city != '':
        print(f"Price from Tool [get_ticket_price] for {destination_city} is {price}")
    return price

# There's a particular dictionary structure that's required to describe our function:
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

# We have to write that function handle_tool_call:

def handle_tool_calls(tool_calls: list[ChatCompletionMessageToolCall]):
    city = ''
    price = 'Unknown'
    tool_message_prompt = {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": None}),
        "tool_call_id": None
    }

    for tool_call in tool_calls:
        tool_call_id = tool_call.id
        tool_call_type = tool_call.type
        func_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f'func_name: {func_name}')
        print(f'tool_call_type: {tool_call_type}')

        # Get Ticket Price
        if func_name == 'get_ticket_price' and tool_call_type == 'function':
            city = arguments.get('destination_city')
            price = get_ticket_price(city)
            tool_message_prompt = {
                "role": "tool",
                "content": json.dumps({"destination_city": city, "price": price}),
                "tool_call_id": tool_call_id
            }
            print(f'tool_message_prompt: {tool_message_prompt}')
            return tool_message_prompt, city, price

    return tool_message_prompt, city, price

# And this is included in a list of tools:
tools = [{"type": "function", "function": price_function}]

####################### END - TOOLS ######################

######################### HELPER #########################
def artist(city, provider: str='StableDiffusion', size: str='512x512'):
    prompt = f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style"

    image_generator: ImageGenerator = ImageGenerator(provider=provider, base_url='http://127.0.0.1:7860')
    size = '1024x1024' if provider == 'GPT' else size
    generated_image = image_generator.generate(prompt=prompt, size=size)
    # generated_image.show()
    save_path = f"images/new_artist_{time.time()}.png"
    generated_image.save(save_path)
    print(f'A new imaged generated at {save_path}.')
    return generated_image

# artist('New York City')
# artist('Ho Chi Minh City')


def talker(message):
    audio = Audio()
    audio.read_from_text(message)

# talker("Well, hi there, baby")
###################### END - HELPER ######################

################## HANDLE SUBMIT BUTTON ##################
open_chat = OpenChat(model='llama3.2', api_key='ollama', call_type='openai')
def chat(conversation_histories: list):
    system_prompt = "You are a helpful assistant for an Airline called FlightAI. "
    system_prompt += "Give short, courteous answers, no more than 1 sentence. "
    system_prompt += "Always be accurate. If you don't know the answer, say so."
    print(f'conversation_histories: {conversation_histories}')
    messages = [{"role": "system", "content": system_prompt}] + conversation_histories

    response: ResponseChat = open_chat.send2(messages=messages, tools=tools)
    image = None

    if response.origin_response.choices[0].finish_reason == "tool_calls":
        message_object = response.origin_response.choices[0].message
        print(f'message_object: {message_object}')
        tool_message_prompt, city, price = handle_tool_calls(message_object.tool_calls)
        messages.append(message_object)
        messages.append(tool_message_prompt)

        # Generate image from text (about city)
        if price != 'Unknown':
            # image = artist(city, provider='GPT')
            image = artist(city, provider='StableDiffusion', size='256x256')

        response = open_chat.send2(messages=messages)

    reply = response.origin_response.choices[0].message.content
    conversation_histories += [{"role": "assistant", "content": reply}]

    print(f'Reply ===> {reply}')

    # Read content of reply
    talker(reply)

    return conversation_histories, image

############### END - HANDLE SUBMIT BUTTON ###############


# Start server
with gr.Blocks() as ui:
    with gr.Row(): # Row 1: 1 place for chatbot messages, other palce for showing image if found
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row(): # Row 2:  1 textinput for enter your message
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row(): # row 3: Clear bbutton
        clear = gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role": "user", "content": message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)

