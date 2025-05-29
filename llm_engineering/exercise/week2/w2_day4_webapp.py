import json
import os
import sys
from typing import List

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

########################## TOOLS #########################
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

# Useful function
def get_ticket_price(destination_city):
    destination_city = '' if destination_city == None else destination_city.strip()
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

        # Get Ticket Price
        if func_name == 'get_ticket_price' and tool_call_type == 'function':
            city = arguments.get('destination_city')
            price = get_ticket_price(city)
            tool_message_prompt = {
                "role": "tool",
                "content": json.dumps({"destination_city": city, "price": price}),
                "tool_call_id": tool_call_id
            }
            return tool_message_prompt, city

    return tool_message_prompt, city

# And this is included in a list of tools:
tools = [{"type": "function", "function": price_function}]

####################### END - TOOLS ######################

################## HANDLE SUBMIT BUTTON ##################
chat = OpenChat(model='llama3.2', api_key='ollama', call_type='openai')
def chatbot(message: str, history: list):
    system_prompt = "You are a helpful assistant for an Airline called FlightAI. "
    system_prompt += "Give short, courteous answers, no more than 1 sentence. "
    system_prompt += "Always be accurate. If you don't know the answer, say so."

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]

    response: ResponseChat = chat.send2(messages=messages, tools=tools)

    if response.origin_response.choices[0].finish_reason == "tool_calls":
        message = response.origin_response.choices[0].message
        print(f'message: {message}')
        tool_message_prompt, city = handle_tool_calls(message.tool_calls)
        messages.append(message)
        messages.append(tool_message_prompt)
        response = chat.send2(messages=messages)

    content = response.origin_response.choices[0].message.content
    print(f'Answer ===> {content}')
    return content

############### END - HANDLE SUBMIT BUTTON ###############


web = gr.ChatInterface(
    fn=chatbot,
    type='messages',
)

# 2. Start server
web.launch(
    server_port=7861,
    share=False, # True: share via internet (https://a23dsf231adb.gradio.live), False: local
    # inbrowser=True # open new tab automatically
)
