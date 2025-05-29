import os
import sys

# Current fdir
script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
# Append the relative paths
sys.path.append("..")

from LLM import LLM
from enums.ServiceEnum import ServiceEnum


# llm = LLM(service=ServiceEnum.OLLAMA)
llm = LLM(service=ServiceEnum.GEMIMI)
# llm = LLM(service=ServiceEnum.DEEPSEEK)
# llm = LLM(service=ServiceEnum.CLAUDE)
# llm = LLM(service=ServiceEnum.GPT)

system_prompt = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

response = llm.predict(system_prompt=system_prompt, user_prompt=user_prompt)

print(response)