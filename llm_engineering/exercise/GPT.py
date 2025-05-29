import os

from dotenv import load_dotenv
from interface import implements
from openai import OpenAI
from AIChat import AIChat
from Util import Util

load_dotenv(override=True)


# class GPT(AIChat):
class GPT(implements(AIChat)):

    __api_key: str = None
    __model: str = 'gpt-4.1'
    __client: OpenAI = None

    def __init__(self, model:str=None, api_key: str=None):
        self.__api_key = api_key if Util.not_empty(api_key) else os.getenv('OPENAI_API_KEY', self.__api_key)
        if Util.not_empty(model):
            self.__model = model
        if self.__client is None:
            self.__client = OpenAI(api_key=self.__api_key)
        super().__init__()

    def predict(self,
             system_prompt: str,
             user_prompt: str=None,
             temperature: float=None,
             response_format: str='text',
             stream: bool=False):


        """_summary_

        Args:
            system_prompt (str): System message
            user_prompt (str, optional): User  message. Defaults to None.
            temperature (float, optional): _description_. Defaults to 0.7.
            response_format (str, optional): Response format (text, json_object, json_scheme). Defaults to 'text'.

        Returns:
            _type_: _description_
        """
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        if system_prompt is not None and system_prompt != '':
            messages.append({"role": "system", "content": system_prompt})

        response = self.__client.chat.completions.create(model=self.__model,
                                                         messages=messages,
                                                         temperature=temperature,
                                                         response_format={'type': response_format},
                                                         stream=stream)

        # If streamed
        if stream:
            return response

        return response.choices[0].message.content
