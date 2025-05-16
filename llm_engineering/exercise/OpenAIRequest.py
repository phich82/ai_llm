import os
from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv


load_dotenv(override=True)


class OpenAIRequest:

    __openai: OpenAI

    __BASE_URL: str = 'http://localhost:11434/v1'
    __API_KEY: str = os.getenv('OPENAI_API_KEY', 'ollama')
    __MODEL: str = 'llama3.2'

    def __init__(self, api_key: str=None, model: str=None):
        if api_key != None and api_key != '':
            self.__API_KEY = api_key

        if model != None and model != '':
            self.__MODEL = model

        self.__openai = OpenAI(base_url=self.__BASE_URL, api_key=self.__API_KEY)

    def create(self, message: str, role: str='user') -> ChatCompletion:
        """_summary_

        Args:
            message (str): _description_
            role (str, optional): _description_. Defaults to 'user'.

        Returns:
            ChatCompletion: _description_
        """

        response = self.__openai.chat.completions.create(model=self.__MODEL, messages=[{"role": role, "content": message}])
        return response

    def create_user_and_system(self, message_user: str,
                               message_system: str,
                               role_user: str='user',
                               role_system: str='system') -> ChatCompletion:
        """_summary_

        Args:
            message_user (str): _description_
            message_system (str): _description_
            role_user (str, optional): _description_. Defaults to 'user'.
            role_system (str, optional): _description_. Defaults to 'system'.

        Returns:
            ChatCompletion: _description_
        """

        messages = [
            {"role": role_system, "content": message_system},
            {"role": role_user, "content": message_user}
        ]
        print(f'messages: {messages}')
        response = self.__openai.chat.completions.create(
            model=self.__MODEL,
            messages=messages
        )
        return response
