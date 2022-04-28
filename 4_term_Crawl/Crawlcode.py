#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import time
import traceback
import openpyxl

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

from bs4 import BeautifulSoup
import requests


# In[2]:


options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])


# In[23]:


driver = webdriver.Chrome('C:/selenium/chromedriver.exe')
driver.get('https://everytime.kr/login')
driver.implicitly_wait(1)

driver.find_element_by_name('userid').send_keys('###아이디 입력해')
driver.find_element_by_name('password').send_keys('###비밀번호입력해')


driver.find_element_by_xpath('/html/body/div/form/p[3]/input').click()
driver.implicitly_wait(1)


# In[48]:


keyword = input('검색어 입력해주세요!!')

driver.find_element_by_xpath('/html/body/div[2]/div[3]/form/input').send_keys(keyword)
driver.find_element_by_xpath('/html/body/div[2]/div[3]/form/input').send_keys(Keys.RETURN)


# In[34]:


#driver.get('https://everytime.kr')


# In[37]:


date = []
title = []
context = []
comment = []

### 최대로 몇페이지 크롤링할까요?ㅎㅎㅎ
max_page = int(input("몇페이지할까요?"))
for page in range(max_page):
    for text_num in range(20):
        text_xpath = "/html/body/div[2]/div[2]/article[" + str(text_num + 1) + "]/a"
        driver.find_element_by_xpath(text_xpath).click()
        
        try:
            date_elem = driver.find_element_by_xpath('/html/body/div[2]/div[2]/article/a/div[1]/time').text
            title_elem = driver.find_element_by_xpath('/html/body/div[2]/div[2]/article/a/h2').text
            context_elem = driver.find_element_by_xpath('/html/body/div[2]/div[2]/article/a/p').text
            comment_elem = driver.find_element_by_xpath('/html/body/div[2]/div[2]/article/div').text
        
            date.append(date_elem)
            title.append(title_elem)
            context.append(context_elem)
            comment.append(comment_elem)
        except:
            print("제목없는글")
        
        driver.back()
    driver.find_element_by_class_name('next').click()   


# In[38]:


df_dic = {
    '제목': title,
    '내용': context,
    '날짜': date,
    '댓글': comment
}

df = pd.DataFrame(df_dic)


# In[40]:


df


# In[44]:


# 올해는 날짜에 연도가 없어서 날짜에 연도를 추가해주는 코드야

for i in range(len(df)):
    if len(df['날짜'][i]) < 12:
        df['날짜'][i] = '22/' + df['날짜'][i]


# In[47]:


df


# In[46]:


df.to_csv("에타_축제_크롤링.csv", encoding='utf-8-sig')


# In[ ]:




