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

#-----------------------------------------------------
df_sample = df_corona.iloc[0:1000, :]
#               f,ax = plt.subplots(1, 1,figsize=(20,10))
#               sns.countplot('언론사', data=df_sample)
#               ax.set_title('언론사 별 낸 기사 개수', y=1.02)
#               plt.xticks(rotation=75)
               
#               #sns.countplot('기고자', data=df_sample, ax=ax[1])
#               #ax[1].set_title('기고자 별 낸 기사 개수', y=1.02)
               
#               print(plt.show())
#-----------------------------------------------------

#평균값 기준 이상값들만 표시 나머지는 others
#print(df_sample.head())

df_group = df_sample.iloc[:,[1,3]].groupby('언론사').count().reset_index()
df_group['제목'].mean() # 여기까지 이제 언론사 중앙값 구한겨

#df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')

for i in df_group['제목'].sort_values(ascending=True):
    if i >= df_group['제목'].mean():
        mlist_dfg = df_group['언론사'][df_group['제목']==i]
        mlist_dfg = mlist_dfg.to_list()
        mlist_dfg.append(i)
        #a = "".join(a) # 만약 str형태로 해야되면
        break
    
df_bfmap = df_sample.iloc[:,[1,3]].groupby('언론사').count().sort_values(by='제목').reset_index()
df_bfmap.columns = ['언론사', '기사개수']


#def map_df(x):
#    others = []
#    for j,v in x['기사개수'], range(len(x['기사개수'])):
#        if j < mlist_dfg[1]:
#            others.append(j)
#        else:
#            others = sum(others) / len(others) * 2 #밸런스 패치를 위해 나머지 기사 개수를 '남은개수 평균 *2' 로 잡고 매핑하자
#    
#            break

#for j,v in df_bfmap['기사개수'], df_bfmap['기사개수'].index():
#    if v <= 5:
#        print(j,v)
#    else:break

#print(df_bfmap.index()) #index나 길이만큼 할당해서 for함수에 각각 집어넣어야 하는데 잘안되네..?

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