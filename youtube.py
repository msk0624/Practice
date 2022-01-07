from selenium import webdriver
from bs4 import BeautifulSoup as bs
import pandas as pd
from selenium.webdriver.common.keys import Keys
import time

keyword = '국내여행'
url = 'https://www.youtube.com/results?search_query={}'.format(keyword)

driver = webdriver.Chrome('./chromedriver.exe')
driver.get(url)
soup = bs(driver.page_source, 'html.parser')
driver.close()

name = soup.select('a#video-title')
video_url = soup.select('a#video-title')
view = soup.select('a#video-title')

name_list = []
url_list = []
view_list = []

for i in range(len(name)):
    name_list.append(name[i].text.strip())
    view_list.append(view[i].get('aria-label').split()[-1])
for i in video_url:
    url_list.append('{}{}'.format('https://www.youtube.com',i.get('href')))
    
youtubeDic = {
    '제목': name_list,
    '주소': url_list,
    '조회수': view_list
}

youtubeDf = pd.DataFrame(youtubeDic)

youtubeDf.to_csv('오마이걸유튜브.csv', encoding='utf-8-sig', index=False)