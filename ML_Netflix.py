# 0. libarary 
import re
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from paramiko import Agent
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree



# 1. percenta,b 결측치를 평균으로 처리

df_train = pd.read_csv('netflix_dataset.csv',encoding='utf-8')
df_train.info()

amean_value = round(df_train['percenta'].mean(),2)
bmean_value = round(df_train['percentb'].mean(),2)

df_train['percenta'].fillna(float(amean_value), inplace=True)
df_train['percentb'].fillna(float(bmean_value), inplace=True)

df_train.dropna(inplace=True)


# 2. 전처리 안할 컬럼 삭제

df_train.columns
df_train.drop(['show_id','type','title','director','description','date_added'],axis=1,inplace=True)


# 3. country에서 적은 개수를 가진 나라들을 모두 기타 처리 후 범주데이터 화.

df_train['country'].value_counts()
df_train['country'][(df_train['country'] != 'United States') & (df_train['country'] != 'India')] = 'etc'
df_train = pd.get_dummies(data=df_train, columns=['country'], dtype=float)


# 4. rating 
# Some are meant for very young children (TV-Y, TV-Y7), -> kids
# some are meant for everyone or mostly everyone (G, PG), -> kids
# and some are meant for older audiences (PG-13, TV-14, R, TV-MA). -> some are teens, others are adults 


df_train['rating'].value_counts()
df_train['rating'][(df_train['rating'] == 'R') | (df_train['rating'] == 'TV-MA')] = 'adults'
df_train['rating'][(df_train['rating'] == 'TV-14') | (df_train['rating'] == 'PG-13')] = 'teens'
df_train['rating'][(df_train['rating'] != 'adults') & (df_train['rating'] != 'teens')] = 'kids'
df_train = pd.get_dummies(data=df_train, columns=['rating'], dtype=float)


# 5. release year 일단 보류

df_train['release_year'].value_counts()


# 6. listed in



# 7. percenta, percentb to reputation / 일다 기준 0.7로 잡음

df_train['percenta'][(df_train['percenta'] >= 0.7)] = 1
df_train['percenta'][(df_train['percenta'] < 0.7)] = 0
df_train['percenta'].value_counts()

df_train['percentb'][(df_train['percentb'] >= 0.7)] = 1
df_train['percentb'][(df_train['percentb'] < 0.7)] = 0
df_train['percentb'].value_counts()

df_train['reputation'] = df_train['percenta'] * df_train['percentb']
df_train['reputation'].value_counts()

# 8. duration
df_train['duration'].value_counts
df_train['duration'] = df_train['duration'].str.extract(r'(\d+)')
df_train['duration'] = df_train['duration'].astype('int')
df_train.info()

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
df_train.drop(['country_etc','rating_adults'],axis=1,inplace=True) # 준거집단 모두 삭제
df_train.drop(['percenta','percentb'],axis=1,inplace=True) # 안쓰는 열 삭제
df_train.drop(['listed_in','cast'], axis=1,inplace=True) # 일단 보류

df_train.columns
df_train.info()

# df_train.to_csv("netflix_dataset_2.csv",encoding='utf-8')



## Logistic Regression & Random Forest
X = df_train.drop('reputation', axis=1)
y = df_train['reputation']                  
                  
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, test_size=0.30, 
                                                    random_state=101)

models=[]
models.append(('LR',LogisticRegression()))
models.append(('RFC',RandomForestClassifier()))

for name, model in models:
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    

rfc = RandomForestClassifier(max_depth = 8, n_estimators=20)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

pd.DataFrame(rfc.feature_importances_, X_train.columns, columns=['Feature Importance'])


# tree.plot_tree(rfc.estimators_[0], feature_names=X.columns, filled=True)
# plt.show()