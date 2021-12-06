# 캐글 분석 프로젝트에 필요한 패키지 불러오기
import os
from keras.engine import sequential
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

#print(df_train['Age'].isnull().sum())
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

#print(df_train.isnull().sum()[df_train.isnull().sum() > 0])
#print(df_test.isnull().sum()[df_test.isnull().sum() > 0])
# #null값이 있는 행들 추출해본 결과 Age의 널값이 지워졌음을 알 수 있다.

#print('Embarked has', sum(df_train['Embarked'].isnull()), 'Null values')
# #Embarked 열에는 null 값이 2개 있는것을 알 수 있음.

df_train['Embarked'].fillna('S', inplace=True)
# #null 값을 'S'으로 대체. fillna가 함수고 inplace는 바로 데이터 프레임에 적용할 수 있게하는 명령어

#print(df_train.isnull().sum())
#print(df_train.isnull().sum()[df_train.isnull().sum()>0])
# #df_train의 널값이 있는 열을 골라서(조건) 그 열의 널값을 다시 sum하면 cabin만 널값이 있음을 알 수 있다.

def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7
# #age mean을 채울때 loc 함수를 썻지만 여기서는 apply함수로 age를 카테고리화 해서 열을 새로 만들어 보겠습니당.

df_train['Age_cat'] = df_train['Age'].apply(category_age)
df_test['Age_cat'] = df_test['Age'].apply(category_age)

#print(df_train.groupby(['Age_cat'])['PassengerId'].count())
# #Age 카테고리화 성공

df_train['Initial'] = df_train['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})
df_test['Initial'] = df_test['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})
# #컴퓨터가 이니셜을 인식할수 있도록 수치화 시킴

df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'Q':1, 'S':2})
df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'Q':1, 'S':2})

#print(df_train['Embarked'].isnull().any(), df_train['Embarked'].dtypes)
# #Embarked에 널값이 없고 데이터 타입이 숫자인걸 알 수 있음.

# #이제 각 feature간 상관관계를 분석할 것임.
# # -1에 가까울수록 반비례, 0에 가까울수록 관계없음, 1에 가까울수록 정비례

df_train['Sex'] = df_train['Sex'].map({'male':0, 'female':1})
df_test['Sex'] = df_test['Sex'].map({'male':0, 'female':1})

#print(df_train.head())

#heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat', 'Age']]
#colormap = plt.cm.RdBu
#plt.figure(figsize=(14,12))
#plt.title('Pearson Correlation Of Features', y=1.05, size=15)
#sns.heatmap(heatmap_data.astype(float).corr(),linewidths=0.1,vmax=1.0,
#            square=True,cmap=colormap,linecolor='white',annot=True, annot_kws={"size":16})
#
#print(plt.show())
#del heatmap_data

# #상관관계를 보니 Sex 와 Pclass가 Survived 와 관계있음.
# #Fare와 Embarked도 어느정도 상관있음을 알 수 있음.
# #서로 강한 상관관계를 가진 feature들이 없음. 이것은 우리가 모델을 학습시킬때 불필요한 featrure가 없음을 의미.
# #만약 1 또는 -1 관계를 가진 feature들이 있으면 우리가 얻을 수 있는 정보는 하나로 확정되기 때문!!

# #데이터 전처리 작업 시작

df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
# #모델의 성능을 높이기 위해 벡터형 데이터로 바꾸어주는 'One-hot Encoding' 방식을 사용
# #이렇게 함으로써 각 클래스간 연관성을 Orthogonal(직교, 동일하게) 만들 수 있음

df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
# #Embarked 열도 동일하게 적용

#print(df_train.head())
# #만약 'One-hot Encoding'을 했을때 category가 100개가 넘어가는 경우가 생김. 이러면 학습시 매우 버거움. 이것을 '차원의 저주' 라고 부름!!

df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
# #axis 는 행,열임 0이 행, 1이 열

#print(df_train.head())
#print(df_train.dtypes)
#print(df_test.head())
#print(df_test.dtypes)

# #머신러닝을 시작하겠다! Target class가 Survived임. 이것은 0과 1로 이루어진 binary classfication 문제임.
# #train set의 survived를 제외한 각 열을 가지고, 모델을 최적화 시켜서, 각 샘플(탑승객)의 생존유무를 판다하는 모델을 만들 것이다.
# #그 후 모델이 학습하지 않았던 test set을 줘서 여기의 샘플의 생존유무를 예측해볼 것이다.

from sklearn.ensemble import RandomForestClassifier #유명한 RamdonForest..
from sklearn import metrics #모델의 평가를 위해 씀
from sklearn.model_selection import train_test_split #training을 쉽게 나눠주는 함수.

X_train = df_train.drop('Survived',axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values

#print(X_train.shape, X_test.shape)

# #우리는 더 좋은 모델을 만들기 위해서 valid(dev) set도 만들거임.
# #비유를 하면 공부(train)하고 모의고사(valid)를 치고 수능(test)를 보는 느낌으로.

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.2, random_state=2018)
#print(X_tr.shape, X_vld.shape, y_tr.shape, y_vld.shape)
# # X_train을 80%, 20%로 나누고, target_label 도 마찬가지로 나눔.

model = RandomForestClassifier() #모델 생성
model.fit(X_tr, y_tr) #학습
prediction = model.predict(X_vld) #예측

#print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
# #아무런 파라미터 튜닝 없이 정확도가 80퍼 가까이 나옴!!

# #학습된 모델은 feature importance를 가지고 있습니당.
# #쉽게 말해 y = 4x1 + 2x2 라고 하면 우리는 X1이 y에 더 큰 영향을 미친다는 것을 알고 있습니다.
# #이 학습된 모델을 pandas를 이용하면 쉽게 분류하여 그래프를 그릴 수 있습니다.

from pandas import Series
feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)

#colors = sns.color_palette('hls',len(Series_feat_imp.sort_values())) 
#개수만큼 색상 설정

#plt.figure(figsize=(8,8))
#Series_feat_imp.sort_values(ascending=True).plot.barh(color = colors)
#plt.xlabel('Feature importance')
#plt.ylabel('Feature')
#print(plt.show())

# #Keras를 이용한 NN모델 개발
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD

nn_model = Sequential()
nn_model.add(Dense(32,activation='relu',input_shape=(14,)))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(64,activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(32,activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1,activation='sigmoid'))

Loss = 'binary_crossentropy'
nn_model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])
#print(nn_model.summary())

#history = nn_model.fit(X_tr,y_tr,
#                    batch_size=64,
#                    epochs=10,
#                    validation_data=(X_vld, y_vld),
#                    verbose=1)

#hists = [history]
#hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)
#hist_df.index = np.arange(1, len(hist_df)+1)
#fig , axs = plt.subplots(nrows=2,sharex=True,figsize=(16,10))
#axs[0].plot(hist_df.val_accuracy, lw=5, label='Validation Accuracy')  #lw는 선의굵기
#axs[0].plot(hist_df.accuracy, lw=5, label='Training Accuracy')
#axs[0].set_ylabel('Accuracy')
#axs[0].set_xlabel('Epoch')
#axs[0].grid()
#axs[0].legend(loc=0)
#axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')
#axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')
#axs[1].set_ylabel('MLogLoss')
#axs[1].set_xlabel('Epoch')
#axs[1].grid()
#axs[1].legend(loc=0)
#fig.savefig('hist.png') #이미지 파일로 저장
#os.getcwd() #파이썬 작업 경로 확인(저장, 및 불러오는 곳)
#print(plt.show())

# #모델 예측 및 평가

#submission = pd.read_csv('./gender_submission.csv')
##print(submission.head())

#prediction = model.predict(X_test)
#submission['Survived'] = prediction
##submission.to_csv('my_first_submission.csv', index=False) #예측 파일 생성 후 비교

#prediction = nn_model.predict(X_test)
#prediction = prediction > 0.5
#prediction = prediction.astype(np.int)
#prediction = prediction.T[0]
#submission['Survived'] = prediction
#print(prediction.shape)
##submission.to_csv('my_first_submission.csv', index=False) #예측 파일 생성 후 비교

## 이로써 처음 해본 titanic 생존 예측 텀과제를 마무리하겠씁니다!!!  