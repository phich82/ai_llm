import os
from dotenv import load_dotenv
import google.generativeai
from interface import implements
from openai import OpenAI
from AIChat import AIChat
from Util import Util

load_dotenv(override=True)


# class Gemini(AIChat):
class Gemini(implements(AIChat)):
    """Class represnting a gemini LLM from Google"""

    __base_url: str = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    __api_key: str = None
    __model: str = 'gemini-2.0-flash'
    __use_openai: bool = True
    __client: OpenAI = None

    def __init__(self, model: str=None, api_key: str=None, use_openai: bool=True):
        self.__api_key = api_key if Util.not_empty(api_key) else os.getenv('GOOGLE_API_KEY', self.__api_key)
        self.__use_openai = use_openai
        if Util.not_empty(model):
            self.__model = model
        if not self.__use_openai:
            google.generativeai.configure(api_key=self.__api_key)
        else:
            if self.__client is None:
                self.__client = OpenAI(base_url=self.__base_url, api_key=self.__api_key)
        super().__init__()

    def predict(self,
             system_prompt: str,
             user_prompt: str=None,
             temperature: float=None,
             response_format: str='text',
             stream: bool=False):
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "system", "content": system_prompt},
        ]

        # 1. Use OpenAI
        if self.__use_openai:
            response = self.__client.chat.completions.create(
                model=self.__model,
                messages=messages,
                temperature=temperature,
                response_format={'type': response_format},
                stream=stream
            )

            # If steaamed
            if stream:
                return response

            return response.choices[0].message.content

        # 2. Use Gemini
        gemini = google.generativeai.GenerativeModel(
            model_name=self.__model,
            system_instruction=system_prompt
        )
        response = gemini.generate_content(user_prompt, stream=stream)

         # If steaamed
        if stream:
            return response

        return response.text


