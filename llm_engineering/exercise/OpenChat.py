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
    __CHAT_API: str = "http://localhost:11434/api/chat"
    __HEADERS: dict = {"Content-Type": "application/json"}
    __MODEL: str = "llama3.2"
    __CALL_TYPE: str = 'http' # http | ollama | openai
    __API_KEY: str = 'ollama'
    __IGNORE_BASE_URL: bool = False

    def __init__(self,
                 ignore_base_url: bool=False,
                 model: str='llama3.2',
                 api_key: str='ollama',
                 call_type: str='http'):
        self.__MODEL = model
        self.__API_KEY = api_key
        self.__CALL_TYPE = call_type
        self.__IGNORE_BASE_URL = ignore_base_url


    def send(self,
             message: str=None,
             user_prompt: str=None,
             system_prompt: str=None,
             stream: bool=False,
             response_format: str='text') -> ResponseChat:
        """Send questions

        Args:
            message (str): user message
            user_prompt (str): user message
            system_prompt (str): system message
            stream (bool): streamed or not
            response_format (str): text | json_object | json_schema
        """

        messages = [
            # { "role": "user", "content": message }
        ]

        if system_prompt != None:
            messages.append({'role': 'system', 'content': system_prompt})

        if user_prompt != None:
            messages.append({'role': 'user', 'content': user_prompt})
        elif message != None:
            messages.append({'role': 'user', 'content': message})

        print(f'Calling via {self.__CALL_TYPE}')
        print(f'Model used: {self.__MODEL}')
        print(f'Message: {len(messages)}')
        print(f'Ignore base url: {self.__IGNORE_BASE_URL}')

        # 1. Using ollama
        if self.__CALL_TYPE == 'ollama':
            response: ollama.ChatResponse = ollama.chat(model=self.__MODEL, messages=messages, stream=stream)

            # If streamed
            if stream:
                return response

            message_chat: ResponseMessageChat = ResponseMessageChat(
                role=response['message']['role'],
                content=response['message']['content']
            )
            response_chat: ResponseChat = ResponseChat(model=self.__MODEL, message=message_chat)
            return response_chat
        # 2. Using OpenAI
        elif self.__CALL_TYPE == 'openai':
            if self.__IGNORE_BASE_URL:
                openai = OpenAI(api_key=self.__API_KEY)
            else:
                openai = OpenAI(base_url=self.__BASE_URL, api_key=self.__API_KEY)
            response: ChatCompletion = openai.chat.completions.create(
                model=self.__MODEL,
                messages=messages,
                response_format={"type": response_format},
                stream=stream
            )

            # If streamed
            if stream:
                return response

            response_chat: ResponseChat = ResponseChat()

            if response is not None and len(response.choices) > 0:
                response_chat.message = ResponseMessageChat(
                    role=response.choices[0].message.role,
                    content=response.choices[0].message.content
                )
            return response_chat
        # 3. Using http (call direct http)
        elif self.__CALL_TYPE == 'http':
            payload = {
                "model": self.__MODEL,
                "messages": messages,
                "stream": stream
            }
            try:
                response = requests.post(self.__CHAT_API, json=payload, headers=self.__HEADERS, verify=False, timeout=5000).json()

                # If streamed
                if stream:
                    return response

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


