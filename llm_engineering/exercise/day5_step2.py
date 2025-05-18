import os
import json
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI, base_url
from Crawler import Crawler
from Util import Util
from OpenChat import OpenChat, ResponseChat


load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''

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

    # MODEL = 'gpt-4.1' # 'gpt-4o-mini'
    # IGNORE_BASE_URL = True
    openchat = OpenChat(ignore_base_url=IGNORE_BASE_URL, model=MODEL, api_key=api_key, call_type='openai')
    response: ResponseChat = openchat.send(user_prompt=user_prompt,
                                           system_prompt=system_prompt,
                                           response_format='json_object')
    result = response.message.content
    return json.loads(result)

# https://edwarddonner.com
# https://huggingface.co
# links = get_links("https://huggingface.co", system_prompt=link_system_prompt, user_prompt_func=get_links_user_prompt)
# print('-----------------------> 1')
# print(links)
# print('-----------------------> 1')

def get_all_details(url):
    result = "Landing page:\n"
    result += Crawler(url).get_contents()
    links = get_links(url, system_prompt=link_system_prompt, user_prompt_func=get_links_user_prompt)
    print("Found links:", links)
    for link in links["links"]:
        try:
            if isinstance(link, dict):
                result += f"\n\n{link['type']}\n"
                result += Crawler(link["url"]).get_contents()
            elif isinstance(link, str):
                result += Crawler(link).get_contents()
            else:
                print(f'Link not supported: {type(link)}')
        except Exception as e:
            print('-------->')
            print(f'Error link ({link["url"]}): {e}')
            print('-------->')
    return result

# print('-----------------------> 2')
# print(get_all_details("https://huggingface.co"))
# print('-----------------------> 2')

# Response in markdown
# system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
# and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
# Include details of company culture, customers and careers/jobs if you have the information."

# Response in json
system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in JSON format.\
Include details of company culture, customers and careers/jobs if you have the information."

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    # Response in markdown fomrat
    # user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    # Response in JSON fomrat
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in JSON format.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

# print('-----------------------> 3')
# get_brochure_user_prompt("HuggingFace", "https://huggingface.co")
# print('-----------------------> 3')

def create_brochure(company_name, url):
    # MODEL = 'gpt-4.1' # 'gpt-4o-mini'
    # IGNORE_BASE_URL = True
    openchat = OpenChat(ignore_base_url=IGNORE_BASE_URL, model=MODEL, api_key=api_key, call_type='openai')
    user_prompt = get_brochure_user_prompt(company_name, url)
    response: ResponseChat = openchat.send(user_prompt=user_prompt,
                                           system_prompt=system_prompt,
                                           response_format='text')
    result = response.message.content
    # display(Markdown(result))
    print(result)

# print('-----------------------> 4')
# create_brochure("HuggingFace", "https://huggingface.co")
# print('-----------------------> 4')

def stream_brochure(company_name, url):
    # MODEL = 'gpt-4.1' # 'gpt-4o-mini'
    # IGNORE_BASE_URL = True
    openchat = OpenChat(ignore_base_url=IGNORE_BASE_URL, model=MODEL, api_key=api_key, call_type='openai')
    user_prompt = get_brochure_user_prompt(company_name, url)
    # Do stream here
    stream = openchat.send(user_prompt=user_prompt,
                           system_prompt=system_prompt,
                           response_format='text',
                           stream=True)
    response = ""
    # display_handle = display(Markdown(""), display_id=True)
    i=1
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        response = response.replace("```","").replace("markdown", "")
        # update_display(Markdown(response), display_id=display_handle.display_id)
        print(f'---> {i}')
        print(response)
        print(f'---> {i}')
        i = i+1

print('-----------------------> 5')
stream_brochure("HuggingFace", "https://huggingface.co")
print('-----------------------> 5')