import os
from keras.engine import sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

pd.set_option("display.max_rows", None, "display.max_columns", None)
df_corona = pd.read_csv("./20210731_bigdata_corona.csv") #100mb 이상이어서 올릴려고 git lfs로 깃허브에 올림
df_corona.drop(["주소","일자","원본주소"],axis=1,inplace=True)
#print(df_corona.head())
#print(df_corona.columns)


#음 일단 언론사, 기고자별로 분류 하는 거 해야 될 거 같고.

df_sample = df_corona.iloc[0:1000, :]
#   f,ax = plt.subplots(1, 1,figsize=(20,10))
#   sns.countplot('언론사', data=df_sample)
#   ax.set_title('언론사 별 낸 기사 개수', y=1.02)
#   plt.xticks(rotation=75)
#   
#   #sns.countplot('기고자', data=df_sample, ax=ax[1])
#   #ax[1].set_title('기고자 별 낸 기사 개수', y=1.02)
#   
#   print(plt.show())

# df_sample['count_press'] = df_sample 
#       언론사 별로 count해서 랭크 10개 빼고 나머지 others로 넣고 그래프로 그리고 싶음...
#   df_sample['rank'] = df_sample['카운트 개수'].rank(method='max')

#sns.countplot('기고자', hue='Survived', data=df_corona, ax=ax[1])
#ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)


#통합 분류 1,2를 합치든지 뭘하든지 해서 Nan값 없애야 할 거 같고

#사건 사고 분류 nan값 엄청 많아서 자세히 보고 빼든지, 새로 값 추가 하든지 해야할듯?
#print(df_corona["사건_사고 분류1"].unique())
#print(df_corona["사건_사고 분류2"].unique())
#print(df_corona["사건_사고 분류3"].unique())
#일단 개많네