import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI, base_url
from Crawler import Crawler
from Util import Util
from OpenChat import OpenChat, ResponseChat


load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

MODEL = 'llama3.2' #'gpt-4.1' # 'gpt-4o-mini'
IGNORE_BASE_URL = False

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")

link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


def get_links(url: str, system_prompt: str, user_prompt: str=None, user_prompt_func=None):
    website = Crawler(url)
    user_prompt = user_prompt_func(website) if Util.is_fn(user_prompt_func) else user_prompt

    # MODEL = 'llama3.2' #'gpt-4.1' # 'gpt-4o-mini'
    # IGNORE_BASE_URL = False
    openchat = OpenChat(ignore_base_url=IGNORE_BASE_URL, model=MODEL, api_key=api_key, call_type='openai')
    response: ResponseChat = openchat.send(user_prompt=user_prompt,
                                           system_prompt=system_prompt,
                                           response_format='json_object')
    result = response.message.content
    return json.loads(result)

ed = Crawler("https://edwarddonner.com")
print(ed.links)

huggingface = Crawler("https://huggingface.co")
print(huggingface.links)

links = get_links("https://huggingface.co", system_prompt=link_system_prompt, user_prompt_func=get_links_user_prompt)

print('-----------------------')
print(links)
print('-----------------------')