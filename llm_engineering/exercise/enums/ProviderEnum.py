from enum import Enum, unique


@unique
class ProviderEnum(Enum):
    CHROMA = 'Chroma'
    FAISS = 'Faiss'
    OPENAI = 'OpenAI'
    HUGGING_FACE = 'Hugging Face'

    @classmethod
    def from_name(cls, name: str):
        for (service_name, service) in cls._member_map_.items():
            if service_name == name:
                return service
        raise Exception(f'Name not found: {name}')

    @classmethod
    def get_names(cls):
        return cls._member_names_
