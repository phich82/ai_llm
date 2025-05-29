from enum import Enum, unique


@unique
class ServiceEnum(Enum):
    OLLAMA = 'ollama'
    GPT = 'gpt'
    GEMIMI = 'gemini'
    CLAUDE = 'claude'
    DEEPSEEK = 'deepseek'

    @classmethod
    def from_name(cls, name: str):
        for (service_name, service) in cls._member_map_.items():
            if service_name == name:
                return service
        raise Exception(f'Name not found: {name}')

    @classmethod
    def get_names(cls):
        return cls._member_names_
