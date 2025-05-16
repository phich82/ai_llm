from openai.types.chat import ChatCompletion
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

class Crawler:

    __headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }

    url: str = ''
    title: str = ''
    text: str = ''

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=self.__headers)
        web = BeautifulSoup(response.content, 'html.parser')
        self.title = web.title.string if web.title else "No title found"
        # Ignore tags (script, style, img, input)
        for irrelevant in web.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = web.body.get_text(separator="\n", strip=True)