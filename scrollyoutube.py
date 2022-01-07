import pandas as pd
import time
import traceback
import openpyxl

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
# from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

keyword = '국내여행 브이로그'
url = 'https://www.youtube.com/results?search_query={}'.format(keyword)

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('./chromedriver.exe', options=options)

driver.get(url)
soup = bs(driver.page_source, 'html.parser')

name = soup.select('a#video-title')
video_url = soup.select('a#video-title')
view = soup.select('a#video-title')

name_list = []
url_list = []
view_list = []

#webdriver Scroll Down! 
time.sleep(0.1)
driver.execute_script("window.scrollTo(0, 8000)")
no_of_pagedowns = 10

# print("Scrolling Down!")
while no_of_pagedowns:
    print(10 - no_of_pagedowns, "th Scroll")
    
    for i in range(len(name)):
        name_list.append(name[i].text.strip())
        view_list.append(view[i].get('aria-label').split()[-1])
    for i in video_url:
        url_list.append('{}{}'.format('https://www.youtube.com',i.get('href')))
    
    time.sleep(0.5)
    no_of_pagedowns -= 1
    
youtubeDic = {
    '제목': name_list,
    '주소': url_list,
    '조회수': view_list
}       


youtubeDf = pd.DataFrame(youtubeDic)

youtubeDf.to_csv('국내여행유튜브.csv', encoding='utf-8-sig', index=False)
