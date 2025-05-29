# from abc import ABC, abstractmethod

from interface import Interface

# class AIChat(ABC):

#     @abstractmethod
#     def predict(self,
#                 system_prompt: str,
#                 user_prompt: str=None,
#                 temperature: float=0.7,
#                 response_format: str='text',
#                 stream: bool=False):

#         pass

class AIChat(Interface):

    def predict(self,
                system_prompt: str,
                user_prompt: str=None,
                temperature: float=None,
                response_format: str='text',
                stream: bool=False):
        pass

    # def chat(self,
    #             messages: list,
    #             temperature: float=None,
    #             response_format: str='text',
    #             stream: bool=False):
    #     pass
