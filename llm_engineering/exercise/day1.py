from IPython.core.display import Markdown
from OpenAIRequest import OpenAIRequest
from Crawler import Crawler
from ProgressBar import ProgressBar


if __name__ == '__main__':
    model = 'llama3.2' # gpt-4o-mini | llama3.2
    api_key = 'ollama'
    # Init  Ollama
    ollamaRequest: OpenAIRequest = OpenAIRequest(model=model, api_key=api_key)

    # Crawl the given website
    web = Crawler("https://edwarddonner.com")
    # print(web.title)
    # print(web.text)

    # Define system prompt
    # Exercise: try changing the last sentence to 'Respond in markdown in Spanish or html."
    system_prompt = "You are an assistant that analyzes the contents of a website \
    and provides a short summary, ignoring text that might be navigation related. \
    Respond in html." #markdown

    # A function that writes a User Prompt that asks for summaries of websites:
    # Exercise: try changing the 'summary of this website in markdown 'sentence to
    #           'summary of this website in html."
    def user_prompt_for(website):
        user_prompt = f"You are looking at a website titled {website.title}"
        user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in html. \
If it includes news or announcements, then summarize these too.\n\n"
        user_prompt += website.text
        return user_prompt

    bar = ProgressBar()
    def summarize(url):
        print(url)
        website = Crawler(url)
        response = ollamaRequest.create_user_and_system(message_user=user_prompt_for(website), message_system=system_prompt)
        bar.done()
        response = response.choices[0].message.content
        print(response)
        return response

    bar.start(target=summarize, args=['https://edwarddonner.com'])
    # print(summarize("https://edwarddonner.com"))
