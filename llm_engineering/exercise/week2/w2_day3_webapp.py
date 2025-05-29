import os
import sys

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
from OpenChat import OpenChat


################## HANDLE SUBMIT BUTTON ##################
def chatbot(message: str, history: list, model: str):
    system_prompt = "You are a helpful assistant"
    system_prompt += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    chat = OpenChat(model='llama3.2', api_key='ollama', call_type='openai')
    stream = chat.send2(messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

############### END - HANDLE SUBMIT BUTTON ###############


web = gr.ChatInterface(
    fn=chatbot,
    type='messages',
    flagging_mode='never', # hide 'Flag' button
)

# 2. Start server
web.launch(
    server_port=7861,
    share=False, # True: share via internet (https://a23dsf231adb.gradio.live), False: local
    # inbrowser=True # open new tab automatically
)
