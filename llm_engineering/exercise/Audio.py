import os
import IPython
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import time
import torch

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from io import BytesIO
from openai import OpenAI
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play
from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
    AutoModelForCausalLM
)


load_dotenv(override=True)


class Audio:

    def __get_device(self, device: str=None):
        if device is not None and device != '':
            return device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def remove_file(self, file_path: str):
        try:
            os.remove(file_path)
        except Exception as e:
            print(e)

    def save_temp_file(self, source: any, temp_path: str=None, file_ext: str='wav', samplerate: int=16000) -> str:
        temp_dir = tempfile.gettempdir()
        default_temp_path = os.path.join(temp_dir, f"temp_audio_{time.time()}.{file_ext}")
        temp_path = temp_path if temp_path is not None and temp_path != '' else default_temp_path
        try:
            # AudioSegment
            if isinstance(source, AudioSegment):
                source.export(temp_path, format=file_ext)
                return temp_path
            # Default (numpy.ndarray)
            sf.write(temp_path, source, samplerate=samplerate)
            return temp_path
        except Exception as e:
            print(e)
        return ''

    def play_audio(self, file_path: str, delete_file: bool=False):
        try:
            time.sleep(3) # Student Dominic found that this was needed. You could also try commenting out to see if not needed on your PC
            subprocess.call([
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-hide_banner",
                file_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            if delete_file:
                self.remove_file(file_path)

    def read_from_text(self, message: str, auto_play: bool=True):
        openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",    # 'onyx', 'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse'
            input=message,
            response_format='wav' # mp3, wav
        )
        audio_stream = BytesIO(response.content)
        audio_source = AudioSegment.from_file(audio_stream, format="wav") # mp3, wav

        # Auto play
        if auto_play:
            temp_audio_path = self.save_temp_file(audio_source)
            self.play_audio(temp_audio_path)

        return audio_source

    def read_from_text_by_pipeline(self, message: str, auto_play: bool=True):
        device = self.__get_device()
        audio_generator = pipeline('text-to-speech', 'microsoft/speecht5_tts', device=device)
        embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split="validation")
        speaker_embeddings = embeddings_dataset[7306]["xvector"]
        speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
        speech = audio_generator(
            message,
            forward_params={'speaker_embeddings': speaker_embeddings}
        )

        source: np.ndarray = speech['audio']
        audio_stream = BytesIO()
        sf.write(audio_stream, source, samplerate=speech['sampling_rate'], format='wav')
        audio_source = AudioSegment.from_file(audio_stream, format="wav") # mp3, wav

        # Auto play
        if auto_play:
            temp_audio_path = self.save_temp_file(audio_source, samplerate=speech['sampling_rate'])
            self.play_audio(temp_audio_path)

        return audio_source

    def audio2text_by_openai(self, audio_file: str, device: str=None, model: str='whisper-1', system_message: str, user_prompt: str):
        AUDIO_MODEL = "whisper-1"
        LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        # Sign in to HuggingFace Hub
        hf_token = os.getenv('HF_TOKEN')
        login(hf_token, add_to_git_credential=True)

        # Sign in to OpenAI using Secrets in Colab
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai = OpenAI(api_key=openai_api_key)

        audio = open(audio_file, "rb")
        transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio, response_format="text")

        user_prompt = user_prompt.replace("{{transcription}}", transcription)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        tokenizer = AutoTokenizer.from_pretrained(LLAMA)
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
        outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

        response = tokenizer.decode(outputs[0])

        return response

    def audio2text_by_huggingface(self, audio_file: str, device: str=None, model: str='openai/whisper-medium'):
        model = 'openai/whisper-medium' if model is None or model == '' else model
        device = self.__get_device(device)
        speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=speech_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device=device,
        )

        # Use the Whisper OpenAI model to convert the Audio to Text
        result = pipe(audio_file, return_timestamps=True)
        transcription = result['text']
        print(result)
        return transcription

audio = Audio()
text = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
# audio.read_from_text(text)
audio.read_from_text_by_pipeline(text)