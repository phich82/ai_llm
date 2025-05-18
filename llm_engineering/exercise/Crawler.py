import ssl
import requests
from bs4 import BeautifulSoup

ssl._create_default_https_context = ssl._create_unverified_context

proxies = {
    # 'http': 'your_http_proxy',
    # 'https': 'your_https_proxy',
}

class Crawler:
    """Crawl a website"""

    __headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }

    url: str = ''
    title: str = ''
    text: str = ''
    body: str = ''

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        # response = requests.get(url, headers=self.__headers, timeout=60) # timeout 60s, timeout=None
        response = requests.get(url, headers=self.__headers, timeout=5000, verify=False)
        self.body = response.content
        web = BeautifulSoup(response.content, 'html.parser')
        self.title = web.title.string if web.title else "No title found"
        if web.body:
            # Ignore tags (script, style, img, input)
            for irrelevant in web.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = web.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in web.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        """Get content of website

        Returns:
            str: Website title and content
        """
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
