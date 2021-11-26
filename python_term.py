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

#print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))
#print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))
#print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
# #탑승객 나이 정보

#fig, ax = plt.subplots(1, 1, figsize = (9,5))
#sns.kdeplot(df_train[df_train['Survived']==1]['Age'], ax=ax)
#sns.kdeplot(df_train[df_train['Survived']==0]['Age'], ax=ax)
#plt.legend(['Survived == 1', 'Survived ==0'])
#print(plt.show())
# #나이에 따른 생존 및 사망 비율 선그래프 

#plt.figure(figsize=(8,6))=
#df_train['Age'][df_train['Pclass']==1].plot(kind='kde')
#df_train['Age'][df_train['Pclass']==2].plot(kind='kde')
#df_train['Age'][df_train['Pclass']==3].plot(kind='kde')
#plt.xlabel('Age')
#plt.title('Age Distribution within classes')
#plt.legend(['1st Class','2nd Class','3rd Class'])
#print(plt.show())
# #클래스가 높아질수록 연령대가 올라가네!

#commulate_survival_ratio =[]
#for i in range(1,80):
#    commulate_survival_ratio.append(df_train[df_train['Age']<i]['Survived'].sum() / len(df_train[df_train['Age']<i]['Survived']))

#plt.figure(figsize=(10,10))
#plt.plot(commulate_survival_ratio)
#plt.title('Survival rate change depending on range of Age', y=1.02)
#plt.ylabel('Survival rate')
#plt.xlabel('Range of Age(0~X)')
#print(plt.show())
# #연령별 생존 누적 확률 시각화

# #분석 종합
# 1. 여자
# 2. 어린 나이
# 3. 높은 클래스(1>2>3)
# 위의 세가지 기분에 따라 생존확률이 높다.

#df_train['Embarked'].unique() #항구의 고유값들 나타내기
#f, ax = plt.subplots(1,1, figsize=(10,10))
#df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax)
#print(plt.show())
# #탑승한 항구별 생존율 바그래프

#f, ax = plt.subplots(2,2,figsize=(20,15))
#sns.countplot('Embarked', data=df_train, ax=ax[0,0])
#ax[0,0].set_title('(1) No. Of Passengers Boarded')
#sns.countplot('Embarked', hue = 'Sex', data=df_train, ax=ax[0,1])
#ax[0,1].set_title('(2) Male-Female Split for Embarked')
#sns.countplot('Embarked', hue = 'Survived', data=df_train, ax=ax[1,0])
#ax[1,0].set_title('(3) Embarked vs Survived')
#sns.countplot('Embarked', hue = 'Pclass', data=df_train, ax=ax[1,1])
#ax[1,1].set_title('(4) Embarked vs Pclass')
#plt.subplots_adjust(wspace=0.2, hspace=0.5)
#print(plt.show())

# # Figure(1) - S 항구에서 가장 많은 사람 탑승
# # Figure(2) - C와Q 항구의 남녀비율은 비슷, S는 남성이 많음
# # Figure(3) - 생존확률이 S가 낮음. 
# # Figure(4) - Pclass별로 나눠서 보니 C가 생존확률이 높았던건, 높은 클래스가 많이 타서인걸 보여줌.

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
# # 가족 열 생성 (형제 + 부자 지간 + 자신(=1))

#print("Maximum size of Family: ", df_train['FamilySize'].max())
#print("Maximum size of Family: ", df_train['FamilySize'].min())

#f,ax = plt.subplots(1,3,figsize=(40,10))
#sns.countplot('FamilySize', data=df_train, ax=ax[0])
#ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

#sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
#ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)

#df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
#ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)

#print(plt.show())
#plt.subplots_adjust(wspace=0.5, hspace=0.5)
# #가족사이즈에 따른 (1)가족 사이즈 수 비교, (2)가족 사이즈에 따른 생존자 수 비교, (3)가족 사이즈에 따른 생존자 수 평균

#fig, ax = plt.subplots(1,1,figsize=(8,8))
#g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
#g = g.legend(loc='best')
#print(plt.show())
# #요금에 따른 밀집도

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()
# # #test셋에 Fare에 해당하는 데이터중 널값이 하나 존재하는 것을 확인.
# # #test셋에 있는 nan value를 평균값으로 치환
#
#df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)
#df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i>0 else 0)
#
#fig,ax = plt.subplots(1,1,figsize=(8,8))
#g = sns.distplot(df_train['Fare'], color='b', label="Skewness : {:.2f}".format(df_train['Fare'].skew()),ax=ax)
#g = g.legend(loc='best')
#print(plt.show())
# #log값을 Fare에다 취하니, 비대칭성이 많이 사라짐을 볼수 있음.

#print(df_train['Cabin'].isnull().sum() / df_train.shape[0])
# #Cabin의 널 비율이 77퍼정도임. 그래서 제외
#print(df_train.head()[['PassengerId','Cabin']])
# #실제로 Cabin의 상위 5개항목중 널값이 3개나 있는 것을 볼 수 있음.

#print(df_train.head()['Ticket'].value_counts())
# #티켓 넘버는 매우다양함. 이걸 사용하려면 어떤 특징을 찾아서 데이터와 연결하는 그런 창의적 아이디어가 필요함!

## 데이터분석은 null데이터를 어떻게 처리하냐가 정말 중요하다!!

print(df_train['Age'].isnull().sum())
# #Age에도 널데이터가 있음을 보여줌.

# # ([A-Za-z]+)\. 의 표현 뜻은 [A-Za-z]로 되는 문자가 한번 반복(=)되고, \. (문자 .) 이 붙은 형태로 된 것을 ()범위 내에서 뽑아내라는 뜻.
df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')
df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.')

#pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
# #원래 보여야 하는데..? 편집기 오류인가 안보임;; 
#print(df_train['Initial'].unique())

df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

#print(df_train.groupby('Initial').mean())

#df_train.groupby('Initial')['Survived'].mean().plot.bar()
#print(plt.show())

## null 데이터를 지우기 시작할껀데, 통계학을 사용해도 되고, null이 없는 데이터를 기반으로 새 머신러닝 알고리즘을 만들어도 됨,.
## 우리는 통계학방법을 사용하기로 결정. trains에서 통계학 평균을 추출하여 test의 널값에 넣을 예정

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age'] = 33
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'),'Age'] = 36
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age'] = 5
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age'] = 22
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Other'),'Age'] = 46

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age'] = 33
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age'] = 36
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age'] = 5
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age'] = 22
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Other'),'Age'] = 46

print(df_train.isnull().sum()[df_train.isnull().sum() > 0])
print(df_test.isnull().sum()[df_test.isnull().sum() > 0])
# #null값이 있는 행들 추출해본 결과 Age의 널값이 지워졌음을 알 수 있다.