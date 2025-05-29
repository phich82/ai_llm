import os
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

import tempfile
import subprocess
from io import BytesIO
from pydub import AudioSegment
import time

load_dotenv(override=True)

class Audio:

    def play_audio(self, audio_segment):
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_audio.wav")
        try:
            audio_segment.export(temp_path, format="wav")
            time.sleep(3) # Student Dominic found that this was needed. You could also try commenting out to see if not needed on your PC
            subprocess.call([
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-hide_banner",
                temp_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    def read_from_text(self, message):
        openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",    # 'onyx', 'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse'
            input=message
        )

        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        # play(audio) # file permission error
        self.play_audio(audio)


