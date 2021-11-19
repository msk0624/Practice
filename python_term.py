# 캐글 분석 프로젝트에 필요한 패키지 불러오기
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras # 웬지 모르겠는데 얘는 좀 이상함;; 안되다가 갑자기 되네;;;

plt.style.use('seaborn')
sns.set(font_scale = 2.5)

import missingno as msno
import warnings

warnings.filterwarnings('ignore')
plt.show() # %matplotlib inline 대신 사용

os.listdir("./")

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')
df_submit = pd.read_csv('./gender_submission.csv')

#print(df_train.columns) #열 이름 확인
#print(df_submit.columns)

#print(df_train.head(),df_test.head(),df_submit.head())

#print(df_train.dtypes)
# #각 열의 속성 확인

#print(df_train.describe()) 
# #수치 확인
#print(df_test.describe())
# #확인해보니 서로 count 수치가 다름. 결측치가 있을것으로 추정됨.

#print(df_train.isnull().sum()/df_train.shape[0])
#print(df_test.isnull().sum()/df_test.shape[0])
# #Age와 Cabin 열에 결측치의 20%, 80% 가 존재함을 알 수 있음.

#f, ax = plt.subplots(1,2, figsize=(18,8))

#df_train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0],shadow=True)
#ax[0].set_title('Pie plot - Survived')
#ax[0].set_ylabel('')
#sns.countplot('Survived', data=df_train, ax=ax[1])
#ax[1].set_title('Count plot - Survived')

#print(plt.show())
# # 그래프 결과물 출력 코드

#print(df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).count()) 
# #pclass 별 데이터 카운트
#print(df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).sum()) 
# #pclass 별 생존자 수 합

#print(pd.crosstab(df_train['Pclass'],df_train['Survived'],margins=True))
# #pclass 별 데이터, 생존자 크로스탭

#print(df_train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=True).mean())
# #pclass 별 생존율

#df_train[['Pclass', 'Survived']].groupby(["Pclass"],as_index=True).mean().plot.bar()
# #Pclass 별 생존율 바그래프

#f, ax = plt.subplots(1,2,figsize=(18,10))
#df_train[['Sex',"Survived"]].groupby(['Sex'],as_index=True).mean().plot.bar(ax=ax[0])
#ax[0].set_title('Survived vs Sex')
#sns.countplot('Sex', hue='Survived',data=df_train,ax=ax[1])
#ax[1].set_title('Sex : Survived vs Dead')
# #성별 생존율, 성별 생존 및 죽은 인원 그래프

#sns.factorplot('Pclass','Survived',hue='Sex',data=df_train,size=6,aspect=1.5)
# #Pclass별 남,녀 생존율 선그래프