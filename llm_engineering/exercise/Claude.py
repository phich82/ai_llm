import os
import anthropic
from dotenv import load_dotenv
from interface import implements
from AIChat import AIChat
from Util import Util

load_dotenv(override=True)


# class Claude(AIChat):
class Claude(implements(AIChat)):
    """Class representing a LLM from Anthropic"""

    __api_key: str = None
    __model: str = 'claude-3-7-sonnet-latest'
    __client: anthropic.Anthropic = None

    def __init__(self, model:str=None, api_key: str=None):
        self.__api_key = api_key if Util.not_empty(api_key) else os.getenv('ANTHROPIC_API_KEY', self.__api_key)
        if Util.not_empty(model):
            self.__model = model
        if self.__client is None:
            self.__client = anthropic.Anthropic(api_key=self.__api_key)
        super().__init__()

    def predict(self,
             system_prompt: str,
             user_prompt: str=None,
             temperature: float=None,
             response_format: str='text',
             stream: bool=False):
        messages = [
            {"role": "user", "content": user_prompt},
        ]

        # If streamed
        if stream:
            response = self.__client.messages.stream(
                model=self.__model,
                max_tokens=200,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
            )
            return response

        response = self.__client.messages.create(
            model=self.__model,
            max_tokens=200,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )

        return response.content[0].text
