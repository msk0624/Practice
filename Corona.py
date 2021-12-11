import os
from keras.engine import sequential
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from matplotlib import font_manager, rc #밑에 4줄까지 plt.plot 한글화 패치
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

pd.set_option("display.max_rows", None, "display.max_columns", None) #출력 할때 잘리는 거 없앰

df_corona = pd.read_csv("./20210731_bigdata_corona.csv") #100mb 이상이어서 올릴려고 git lfs로 깃허브에 올림
df_corona.drop(["주소","원본주소"],axis=1,inplace=True) #일단 바로 쓸데 없는 열 삭제


#음 일단 언론사, 기고자별로 분류 하는 거 해야 될 거 같고.

df_sample = df_corona.iloc[0:1000, :]
#               f,ax = plt.subplots(1, 1,figsize=(20,10))
#               sns.countplot('언론사', data=df_sample)
#               ax.set_title('언론사 별 낸 기사 개수', y=1.02)
#               plt.xticks(rotation=75)
               
#               #sns.countplot('기고자', data=df_sample, ax=ax[1])
#               #ax[1].set_title('기고자 별 낸 기사 개수', y=1.02)
               
#               print(plt.show())

#Others가 너무 높으면 안되니까 그 기준을 총 개수의 절반 이하로 잡는게 좋지않을까?
#print(df_sample.head())
#print(df_sample.iloc[:,[1,3]].groupby('언론사').count().sort_values(by='제목'))

df_group = df_sample.iloc[:,[1,3]].groupby('언론사').count().reset_index()
df_group['제목'].mean() # 여기까지 이제 언론사 중앙값 구하는 거임...

#df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')

a = []
for i in df_group['제목'].sort_values(ascending=True):
    if i >= df_group['제목'].mean():
        a.append(df_group['언론사'][df_group['제목']==i]) #얘좀 어떻게 해줘 ㅠㅠㅠㅠ 한글만 추출해봐야 하는데... '동아일보'글자만...
        print(a)
        break

#           df_train['Initial'] = df_train['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4}) 이런 느낌으로 매핑하면 될듯?


#print(df_sample['제목'].isnull().sum())

#                       print(df_sample.drop(['본문'],axis=1).head())
#                       df_sample['기고자'].fillna('이름없음', inplace=True)
#                       print(df_sample['기고자'].isnull().sum())




#------------------------------------------------------------------
#통합 분류 1,2를 합치든지 뭘하든지 해서 Nan값 없애야 할 거 같고
#사건 사고 분류 nan값 엄청 많아서 자세히 보고 빼든지, 새로 값 추가 하든지 해야할듯?
#print(df_corona["사건_사고 분류1"].unique())
#print(df_corona["사건_사고 분류2"].unique())
#print(df_corona["사건_사고 분류3"].unique())
#일단 개많네