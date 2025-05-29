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


################## HANDLE SUBMIT BUTTON ##################
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

def predict(user_prompt: str, model: str):
    # llm = LLM(service=ServiceEnum.GEMIMI)
    llm = LLM(service=ServiceEnum.from_name(model))
    response = llm.predict(
        # system_prompt='You are a helpful assistant',
        system_prompt='You are a helpful assistant that responds in markdown',
        user_prompt=user_prompt,
        stream=True
    )
    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ""
        yield result
    return result

def make_brochure_from_website(company_name: str, url: str, model: str):
    print("make_brochure_from_website ==> start")
    system_prompt = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown."
    user_prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    user_prompt += Crawler(url).get_contents()
    llm = LLM(service=ServiceEnum.from_name(model))
    response = llm.predict(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stream=True
    )
    result = ""
    if model == ServiceEnum.CLAUDE.name:
        with response as stream:
            for text in stream.text_stream:
                result += text or ""
                yield result
    else:
        for chunk in response:
            result += chunk.choices[0].delta.content or ""
            yield result
    # return result

############### END - HANDLE SUBMIT BUTTON ###############


# 1. Setup web ui components
# web = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"], # show 2 inputs
#     outputs=["text"],
#     outputs=['text'],
#     flagging_mode='never', # hide 'Flag' button
#     # live=True # refresh response automatically when inputing
# )
# web = gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Textbox(label='Your message', lines=6),
#         gr.Dropdown(ServiceEnum.get_names(), label='Select model')
#     ], # show 2 inputs
#     outputs=[
#         gr.Textbox(label='Response', lines=12),
#     ],
#     flagging_mode='never', # hide 'Flag' button
#     # live=True # refresh automatically
# )
web = gr.Interface(
    fn=make_brochure_from_website,
    inputs=[
        gr.Textbox(label='Company Name'),
        gr.Textbox(label='Landing page URL including http:// or https://'),
        gr.Dropdown(ServiceEnum.get_names(), label='Select model')
    ], # show 2 inputs
    outputs=[
        gr.Markdown(label='Brochure'),
    ],
    flagging_mode='never', # hide 'Flag' button
    # live=True # refresh automatically
)

# 2. Start server
web.launch(
    server_port=7860,
    share=False, # True: share via internet (https://a23dsf231adb.gradio.live), False: local
    # inbrowser=True # open new tab automatically
)
