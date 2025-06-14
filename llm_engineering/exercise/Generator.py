import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc
from enum import Enum, unique



@unique
class ModelEnum(Enum):
    LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    PHI3 = "microsoft/Phi-3-mini-4k-instruct"
    GEMMA2 = "google/gemma-2-2b-it"
    QWEN2 = "Qwen/Qwen2-7B-Instruct" # exercise for you
    MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # If this doesn't fit it your GPU memory, try others from the hub

    @classmethod
    def from_name(cls, name: str):
        for (service_name, service) in cls._member_map_.items():
            if service_name == name:
                return service
        raise Exception(f'Name not found: {name}')

    @classmethod
    def get_names(cls):
        return cls._member_names_

class Generator:
    """Class representing a specified LLM"""

    __model: ModelEnum = ModelEnum.LLAMA
    __quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    def __init__(self,
                 model: ModelEnum=ModelEnum.LLAMA,
                 quant_config: BitsAndBytesConfig=None,
                 hf_token: str=None):
        if not self.__empty(model):
            self.__model = model
        if not self.__empty(quant_config):
            self.__quant_config = quant_config
        self.__login_hugging_face(hf_token=hf_token)

    def __empty(self, value):
        return value is None or value == ''

    def __normalize(self, value, default_value=None):
        return default_value if self.__empty(value) else value

    def __login_hugging_face(self, hf_token: str=None):
        hf_token = self.__normalize(hf_token, os.getenv('HF_TOKEN'))
        if not self.__empty(hf_token):
            login(hf_token, add_to_git_credential=True)

    def __get_device(self, device: str=None):
        if device is not None and device != '':
            return device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate(self, messages, stream: bool=False, max_token: int=80, model: str=None, device: str='cpu', clear_cache: bool=True):
        model = model if model is not None and model != '' else self.__model.value
        device = self.__get_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(device)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            quantization_config=self.__quant_config
        )
        if stream:
            streamer = TextStreamer(tokenizer)
            outputs = model.generate(inputs, max_new_tokens=max_token, streamer=streamer)
        else:
            outputs = model.generate(inputs, max_new_tokens=max_token)
        if clear_cache:
            try:
                del model, inputs, tokenizer, outputs
                if stream:
                    del streamer
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
        return outputs

if __name__ == '__main__':
    # Test model PHI3
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant"},
    #     {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    # ]
    # generator = Generator(model=ModelEnum.PHI3)
    # generator.generate(messages=messages, stream=True)

    # Test model GEMMA2]
    # messages = [
    #     {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    #   ]
    # generator = Generator(model=ModelEnum.GEMMA2)
    # generator.generate(messages=messages, stream=True)
