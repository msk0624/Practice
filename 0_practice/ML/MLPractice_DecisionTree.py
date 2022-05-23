import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
# %matplotlib inline / 주피터 노트북에서

# import different algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get accuracy score
from sklearn.metrics import accuracy_score

# get f1 score
from sklearn.metrics import f1_score


crowdfunding = pd.read_csv('crowdfunding_cleaned.csv')

crowdfunding.head()
crowdfunding.info()
crowdfunding.describe()

#drop the observations that have missing values 
X_list=['ln_goal','ln_duration','facebook_url', 'imdb_url', 'twitter_url', 
        'youtube_url', 'website_url', 'featured', 'enable_drcc', 'enable_payp',
        'all_or_nothing','funding_started_at_DOW_1.0','funding_started_at_DOW_2.0',
        'funding_started_at_DOW_3.0','funding_started_at_DOW_4.0',
        'funding_started_at_DOW_5.0','funding_started_at_DOW_6.0'] #we use 'ln_goal' and 'ln_duration' instead of 'goal'and'duration'
new_crowdfunding=crowdfunding.dropna(axis=0,how='any',subset=X_list) #drop all rows that have any NaN values

#choose the dependent variable and independent variables 
X = new_crowdfunding[X_list]
y = new_crowdfunding['win']

X.head()
y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

pd.DataFrame(rfc.feature_importances_, X_train.columns, columns=['Feature Importance'])
# 얘가 이 설명변수가 다른 설명변수에 비해 얼마나 중요한지.

# prepare models
models=[]
models.append(('LR',LogisticRegression()))
models.append(('RFC',RandomForestClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))

# evaluate each model in turn
for name, model in models:
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    ac_score=accuracy_score(y_test,predictions)
    msg="%s: %f" % (name, ac_score)
    print(msg)  
    

# evaluate each model in turn
for name, model in models:
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    f_score=f1_score(y_test,predictions)
    msg="%s: %f" % (name, f_score)
    print(msg)
    
# https://velog.io/@ljs7463/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%A0%95%EB%B0%80%EB%8F%84%EC%9E%AC%ED%98%84%EC%9C%A8f1-score%EB%93%B1 참고하면서 공부하면 좋음.