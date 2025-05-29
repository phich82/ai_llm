from AIChat import AIChat
from GPT import GPT
from Claude import Claude
from Gemini import Gemini
from Ollama import Ollama
from DeepSeek import DeepSeek
from enums.ServiceEnum import ServiceEnum


class LLM:
    """Class representing a specified LLM"""

    __service: AIChat = None

    def __init__(self, service: ServiceEnum=ServiceEnum.OLLAMA):
        if self.__service is None:
            self.__service = self.__get_service(service=service)

    def __get_service(self, service: ServiceEnum=ServiceEnum.OLLAMA) -> AIChat:
        print(f'Service ===> {service.name}')
        if service.name == ServiceEnum.GPT.name:
            return GPT()
        if service.name == ServiceEnum.CLAUDE.name:
            return Claude()
        if service.name == ServiceEnum.GEMIMI.name:
            return Gemini(use_openai=True)
        if service.name == ServiceEnum.OLLAMA.name:
            return Ollama()
        if service.name == ServiceEnum.DEEPSEEK.name:
            return DeepSeek()

        raise ValueError(f'Service not support: {service.name}')

    def predict(self,
             system_prompt: str,
             user_prompt: str=None,
             temperature: float=None,
             response_format: str='text',
             stream: bool=False):
        """Predict the answers from user questions

        Args:
            system_prompt (str): System prompt
            user_prompt (str, optional): User prompt. Defaults to None.
            temperature (float, optional): Temperature for depth of focused reasoning. Defaults to None.
            response_format (str, optional): Response format. Defaults to 'text'.
            stream (bool, optional): Streamed or not. Defaults to False.

        Returns:
            dict: Found answers
        """
        return self.__service.predict(system_prompt=system_prompt,
                                      user_prompt=user_prompt,
                                      temperature=temperature,
                                      response_format=response_format,
                                      stream=stream)