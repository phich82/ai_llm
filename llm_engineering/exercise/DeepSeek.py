import os
import anthropic
from dotenv import load_dotenv
from interface import implements
from openai import APIStatusError, OpenAI
from AIChat import AIChat
from Util import Util

load_dotenv(override=True)


# class DeepSeek(AIChat):
class DeepSeek(implements(AIChat)):
    """Class representing a LLM from DeepSeek"""

    __base_url: str = 'https://api.deepseek.com'
    __api_key: str = None
    __model: str = 'deepseek-chat'
    __client: OpenAI = None

    def __init__(self, model:str=None, api_key: str=None):
        self.__api_key = api_key if Util.not_empty(api_key) else os.getenv('DEEPSEEK_API_KEY', self.__api_key)
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
        try:
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
        except Exception as e:
            error_message = f'{e}'
            print(f'DeepSeek:predict => error: {e}')
            if isinstance(e, APIStatusError):
                print(f'Error Message: {e.body["message"]} ({e.status_code})')
                if e.status_code == 402:
                    error_message ='Account amount is not currently enough. You need to deposit money into your account.'
            raise Exception(error_message)

