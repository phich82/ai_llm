from dataclasses import dataclass
from json import JSONDecodeError
from openai import OpenAI
from openai.types.chat import ChatCompletion
import requests
import ollama


@dataclass
class ResponseMessageChat(tuple):
    """_summary_

    Args:
        tuple (_type_): _description_

    Returns:
        _type_: _description_
    """
    role: str
    content: str

    def __new__(cls, role: str=None, content: str=None):
        return tuple.__new__(cls, [role, content])

    def __init__(self, role: str=None, content: str=None):
        tuple.__init__([role, content])
        self.role = role
        self.content = content

@dataclass
class ResponseChat(tuple):
    """_summary_

    Args:
        tuple (_type_): _description_

    Returns:
        _type_: _description_
    """
    model: str
    created_at: str # '2025-05-15T14:58:26.1498237Z'
    done_reason: str
    done: bool
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int
    message: ResponseMessageChat

    def __new__(cls,
                model: str=None,
                created_at: str=None,
                done_reason: str=None,
                done: bool=None,
                total_duration: int=None,
                load_duration: int=None,
                prompt_eval_count: int=None,
                prompt_eval_duration: int=None,
                eval_count: int=None,
                eval_duration: int=None,
                message: ResponseMessageChat=None
                ):
        return tuple.__new__(cls, [
                             model,
                             created_at,
                             done_reason,
                             done,
                             total_duration,
                             load_duration,
                             prompt_eval_count,
                             prompt_eval_duration,
                             eval_count,
                             eval_duration,
                             message
                            ])

    def __init__(self,
                 model: str=None,
                 created_at: str=None,
                 done_reason: str=None,
                 done: bool=None,
                 total_duration: int=None,
                 load_duration: int=None,
                 prompt_eval_count: int=None,
                 prompt_eval_duration: int=None,
                 eval_count: int=None,
                 eval_duration: int=None,
                 message: ResponseMessageChat=None):
        tuple.__init__([model,
                        created_at,
                        done_reason,
                        done,
                        total_duration,
                        load_duration,
                        prompt_eval_count,
                        prompt_eval_duration,
                        eval_count,
                        eval_duration,
                        message
                        ])
        self.model = model
        self.created_at = created_at
        self.done_reason = done_reason
        self.done = done
        self.total_duration = total_duration
        self.load_duration = load_duration
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.message = message

class OpenChat:
    """_summary_

    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """

    __BASE_URL: str = 'http://localhost:11434/v1'
    __OLLAMA_API: str = "http://localhost:11434/api/chat"
    __HEADERS: dict = {"Content-Type": "application/json"}
    __MODEL: str = "llama3.2"
    __CALL_TYPE: str = 'http' # http | ollama | openai
    __API_KEY: str = 'ollama'

    def __init__(self, model: str='llama3.2', api_key: str='ollama', call_type: str='http'):
        self.__MODEL = model
        self.__API_KEY = api_key
        self.__CALL_TYPE = call_type


    def send(self, message: str=None, stream: bool=False) -> ResponseChat:
        """_summary_

        Args:
            payload (Any): _description_
        """

        messages = [
            { "role": "user", "content": message }
        ]

        print(f'Calling via {self.__CALL_TYPE}')
        print(f'Model used: {self.__MODEL}')

        if self.__CALL_TYPE == 'ollama': # Using ollama
            response: ollama.ChatResponse = ollama.chat(model=self.__MODEL, messages=messages, stream=stream)
            message_chat: ResponseMessageChat = ResponseMessageChat(
                role=response['message']['role'],
                content=response['message']['content']
            )
            response_chat: ResponseChat = ResponseChat(model=self.__MODEL, message=message_chat)
            return response_chat
        elif self.__CALL_TYPE == 'openai': # Using OpenAI
            ollama_via_openai = OpenAI(base_url=self.__BASE_URL, api_key=self.__API_KEY)
            response: ChatCompletion = ollama_via_openai.chat.completions.create(
                model=self.__MODEL,
                messages=messages
            )
            response_chat: ResponseChat = ResponseChat()

            if response is not None and len(response.choices) > 0:
                message_chat: ResponseMessageChat = ResponseMessageChat(
                    role=response.choices[0].message.role,
                    content=response.choices[0].message.content
                )
                response_chat.message = message_chat
            return response_chat
        elif self.__CALL_TYPE == 'http': # Using http (call direct http)
            payload = {
                "model": self.__MODEL,
                "messages": messages,
                "stream": stream
            }
            try:
                response = requests.post(self.__OLLAMA_API, json=payload, headers=self.__HEADERS, verify=False, timeout=5000).json()

                message_chat: ResponseMessageChat = ResponseMessageChat(
                    response['message']['role'],
                    response['message']['content']
                )

                response_chat: ResponseChat = ResponseChat(
                    response['model'],
                    response['created_at'],
                    response['done_reason'],
                    response['done'],
                    response['total_duration'],
                    response['load_duration'],
                    response['prompt_eval_count'],
                    response['prompt_eval_duration'],
                    response['eval_count'],
                    response['eval_duration'],
                    message_chat
                )
                return response_chat
            except Exception as e:
                # Wrong UTF codec detected
                if isinstance(e, UnicodeDecodeError):
                    raise Exception('Wrong UTF codec detected.')
                # Catch JSON-related errors
                if  isinstance(e, JSONDecodeError):
                    raise Exception('Invalid json')
                print(f'Exception: {e}')
                raise Exception('Could not parse to json format.')
        raise Exception(f'Call type not support: {self.__CALL_TYPE}')


