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
#print(df_complain['제목'])

#일단 강서구 민원을 통해 주민들이 겪는 어려움에 대해 파악하기 위해서 워드 클라우드 화를 진행
#대게 민원의 구체적인 내용은 제목에 요약되어 있으므로 제목 먼저 진행

stopwords_kr =['및','앞','너무','관련','대한','인한','민원','강서구','후','부탁드립니다','좀','요청','요청합니다','내','대해','바랍니다'
               ,'심합니다','관련해서','올립니다','가는','주세요','강서','강서구청','진짜','이후','이','많이','입니다','따른','왜','수'
               '요청드립니다','어떻게','해주세요','신고합니다','위한','문의드립니다','있습니다','합니다','요청의','문의 드립니다','때문에'
               '없습니다','제발','싶습니다','요청강서구','하는','없는','계속']

def displayWordCloud(data = None,  backgroundcolor = 'black', width=800, height=600 ):
    wordcloud = WordCloud(font_path = font_path, stopwords = stopwords_kr, background_color = backgroundcolor, width = width, height = height).generate(data) 
    print(wordcloud.words_) 
    plt.figure(figsize = (15, 10)) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.show()

#displayWordCloud(''.join(df_complain['제목']))

#워드 클라우드 결과 '공사','소음','도로','설치'에 관한 민원이 가장 많은 것을 알 수 있다.
#더 자세한 분석을 위해 태블로를 통해 민원이 가장 많았던 달에 대해서 분석을 시행해 보겠음

df1 = df_complain[df_complain['제목'].str.contains('공사')]
df1['키워드'] = '공사'
df2 = df_complain[df_complain['제목'].str.contains('소음')]
df2['키워드'] = '소음'
df3 = df_complain[df_complain['제목'].str.contains('도로')]
df3['키워드'] = '도로'
df4 = df_complain[df_complain['제목'].str.contains('설치')]
df4['키워드'] = '설치'

df_main = pd.concat([df1,df2,df3,df4])
df_main.to_csv("강서구_주요민원.csv", encoding='utf-8-sig')