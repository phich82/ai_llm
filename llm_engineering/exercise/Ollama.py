import os
from dotenv import load_dotenv
from interface import implements
from openai import OpenAI

from AIChat import AIChat
from Util import Util

load_dotenv(override=True)


class Ollama(implements(AIChat)):

    __base_url: str = 'http://localhost:11434/v1'
    __api_key: str = 'ollama'
    __model: str = 'llama3.2'
    __client: OpenAI = None

    def __init__(self, model:str=None, api_key: str=None):
        self.__api_key = api_key if Util.not_empty(api_key) else os.getenv('OLLAMA_API_KEY', self.__api_key)
        if Util.not_empty(model):
            self.__model = model
        if self.__client is None:
            self.__client = OpenAI(base_url=self.__base_url, api_key=self.__api_key)
        super().__init__()

    def predict(self,
                system_prompt: str,
                user_prompt: str=None,
                temperature: float=None,
                response_format: str='text',
                stream: bool=False):
        messages = []

        if system_prompt != None:
            messages.append({'role': 'system', 'content': system_prompt})

        if user_prompt != None:
            messages.append({'role': 'user', 'content': user_prompt})

        print(f'Model: {self.__model}')

        response = self.__client.chat.completions.create(
            model=self.__model,
            messages=messages,
            temperature=temperature,
            response_format={"type": response_format},
            stream=stream
        )

        # If streamed
        if stream:
            return response

        return response.choices[0].message.content