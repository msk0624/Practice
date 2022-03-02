#월 별로 content를 키워드로 나누어서 엑셀에 배정한 다음, 그걸 태블로를 통해서 시각화 예정

import os
from keras.engine import sequential
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import collections
from wordcloud import WordCloud, STOPWORDS

#밑에 4줄까지 한글화 패치
from matplotlib import font_manager, rc 
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

df_complain = pd.read_csv("./강서구_민원.csv")
#df = pd.read_csv('bike_demand_train.csv', parse_dates=['datetime'])

df_complain['날짜'] = pd.to_datetime(df_complain['날짜'], format="%Y-%m-%d %H:%M") #object를 datetime형태로 변환
#print(str(df_complain['날짜'])) 

df_complain.set_index('날짜', inplace=True)

df_complain['month']=df_complain.index.month 
df_complain['year']=df_complain.index.year

df_complain_2006 = df_complain.loc['2020-06-01':'2021-06-30']
df_complain_2107 = df_complain.loc['2021-07-01':'2021-07-30']
df_complain_2112 = df_complain.loc['2021-12-01':'2021-12-31']
df_complain_2202 = df_complain.loc['2022-02-01':'2022-02-28']

def displayWordCloud(data = None,  backgroundcolor = 'black', width=800, height=600 ):
    wordcloud = WordCloud(font_path = font_path, background_color = backgroundcolor, width = width, height = height).generate(data) 
    print(wordcloud.words_) 
    plt.figure(figsize = (15, 10)) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.show()

#displayWordCloud(''.join(df_complain_2006['제목']))
#displayWordCloud(''.join(df_complain_2107['제목']))
#displayWordCloud(''.join(df_complain_2112['제목']))
#displayWordCloud(''.join(df_complain_2202['제목']))
#print(df_complain_)

#월별 민원 확인 결과 공사,소음,설치 관련 민원은 2206과 2107에 많은 것을 알 수 있음.

print(df_complain_2006[df_complain_2006['내용'].str.contains('공사')])