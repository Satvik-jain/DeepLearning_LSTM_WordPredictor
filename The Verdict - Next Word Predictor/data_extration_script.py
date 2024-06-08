import requests
from bs4 import BeautifulSoup

URL = "https://en.wikisource.org/wiki/The_Verdict"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

# soup.prettify()

text = [i.text for i in soup.find_all("p")]

text[0]

text = text[0:83]

with open('Data.txt', 'w') as file:
    for string in text:
        file.write(string + '\n')