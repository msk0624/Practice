# 0. libarary 
from cmath import nan
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
from sklearn.preprocessing import StandardScaler


# 1. percenta,b 결측치를 평균으로 처리

df_train = pd.read_csv('netflix_dataset.csv',encoding='utf-8')
df_train.info()

amean_value = round(df_train['percenta'].mean(),2)
bmean_value = round(df_train['percentb'].mean(),2)

df_train['percenta'].fillna(float(amean_value), inplace=True)
df_train['percentb'].fillna(float(bmean_value), inplace=True)

# df_train.dropna(inplace=True)
# df_train.reset_index(inplace=True)
df_train[['percenta','percentb']].fillna(method='pad',inplace=True)


# 2. 전처리 안할 컬럼 삭제

df_train.columns
df_train.drop(['cast','show_id','type','title','director','description','date_added'],axis=1,inplace=True)

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


# 5. release year 은 연속형 변수니 그대로 사용

df_train['release_year'].value_counts()

# 6. listed in
# Dramas, Horor Movies, Comedies, Action, etc
# 정규식으로 앞에 거 있으면 합치는 코드를...짜야하는데...
df_train['listed_in'].value_counts()

df_train['listed_in_first'] = df_train.listed_in.str.split(',').str[0]
df_train['listed_in_first'][(df_train['listed_in_first'] != 'Dramas') & (df_train['listed_in_first'] != 'Comedies')
                            & (df_train['listed_in_first'] != 'Action & Adventure') & (df_train['listed_in_first'] != 'Documentaries')
                            & (df_train['listed_in_first'] != 'Children & Family Movies') & (df_train['listed_in_first'] != 'Horror Movies')
                            ] = 'etc'
df_train['listed_in_first'].value_counts()
df_train = pd.get_dummies(data=df_train, columns=['listed_in_first'], dtype=float)
df_train.drop('listed_in',axis=1,inplace=True)

# 7-1. percenta, percentb를 합친 후 표준화
scaler = StandardScaler()
df_train['percentTotal'] = df_train['percenta'] + df_train['percentb']

# StandardScaler 로 데이터 셋 변환 .fit( ) 과 .transform( ) 호출
scaler.fit(pd.DataFrame(df_train['percentTotal']))
percentTotal_scaled = scaler.transform(pd.DataFrame(df_train['percentTotal']))
# transform( )시 scale 변환된 데이터 셋이 numpy ndarry로 반환되어 이를 DataFrame으로 변환
df_train['Zvalues'] = pd.DataFrame(percentTotal_scaled).values
df_train.info()

# 8. duration
df_train['duration'].value_counts
df_train['duration'] = df_train['duration'].str.extract(r'(\d+)')
df_train['duration'] = df_train['duration'].astype('int')
df_train.info()

# df_train.drop('percentTotal',axis=1, inplace=True)

# df_train.to_csv("netflix_dataset_2.csv",encoding='utf-8')

## Linear Regression
X = df_train.drop(['reputation','Zvalues'], axis=1)
X = df_train.drop(['Zvalues'], axis=1)
y = df_train['Zvalues']                  
                  
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, test_size=0.30, 
                                                    random_state=101)

    
ln = LinearRegression()
ln.fit(X_train, y_train)

coeff_df = pd.DataFrame(ln.coef_,X.columns,columns=['Coefficient'])
coeff_df
# 장르가 선형회귀에서 예측에 중요한 역할을 하는 것을 확인 할 수 있다.

ln_pred = ln.predict(X_test)

plt.scatter(y_test,ln_pred)
plt.show()
print("This model accuracy is {}%".format(round(ln.score(X_test, y_test)*100,2))) 

sns.histplot((y_test-ln_pred),bins=15,kde=True)
plt.show()

y_test1= y_test.to_frame()
y_test1['Predictions'] = ln_pred
y_test1


print('MAE:', metrics.mean_absolute_error(y_test, ln_pred))
print('MSE:', metrics.mean_squared_error(y_test, ln_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ln_pred)))

# 선형회귀로 정확한 점수를 예측하기엔 어려움이 있음을 확인. 그래서 우리는 로지스틱 또는 랜덤포레스트와 같은 분류를 통해서 좋은 평점을 받는다, 아니다의 이분법적인 사고로 정확도를 높이려고 함.

## Logistic Regression & Random Forest


# 7-2. percenta, percentb to reputation / 일단 0.3~0.7의 값 제외

# df_train['percenta'][(df_train['percenta'] >= 0.7)] = 1
# df_train['percenta'][(df_train['percenta'] < 0.7)] = 0
# df_train['percenta'].value_counts()

# df_train['percentb'][(df_train['percentb'] >= 0.7)] = 1
# df_train['percentb'][(df_train['percentb'] < 0.7)] = 0
# df_train['percentb'].value_counts()

# df_train['reputation'] = df_train['percenta'] * df_train['percentb']
df_train['reputation'] = (df_train['percenta'] + df_train['percentb']) / 2
df_train['reputation'].value_counts()
df_train['reputation'][(df_train['reputation'] <= 0.7) & (df_train['reputation'] >= 0.3)] = np.NaN
df_train.dropna(inplace=True)
df_train.info()
df_train['reputation'][(df_train['reputation'] >= 0.7)] = 1
df_train['reputation'][(df_train['reputation'] <= 0.3)] = 0
df_train['reputation'].value_counts()

# df_train['percenta'][(df_train['percenta'] <= 0.7) & (df_train['percenta'] >= 0.3)] = np.NaN
# df_train['percentb'][(df_train['percentb'] <= 0.7) & (df_train['percentb'] >= 0.3)] = np.NaN
# df_train['percenta']


X = df_train.drop(['reputation','Zvalues'], axis=1)
X = df_train.drop(['reputation'], axis=1)
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
    
# 로지스틱 회귀도 긍정적인 예측율을 보임을 알 수 있다. 여기서 랜덤포레스트는 로지스틱에 비해 정확도가 낮은데 이를 더 올려보는 분석을 해보기로 한다.

rfc = RandomForestClassifier(max_depth = 8, n_estimators=24)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

pd.DataFrame(rfc.feature_importances_, X_train.columns, columns=['Feature Importance'])
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

print("This model accuracy is {}%".format(round(rfc.score(X_test, y_test)*100,2))) 

# tree.plot_tree(rfc.estimators_[0], feature_names=X.columns, filled=True)
# plt.show()

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
df_train.drop(['country_United States','rating_adults','listed_in_first_Dramas'],axis=1,inplace=True) # 준거집단 모두 삭제
df_train.drop(['percenta','percentb'],axis=1,inplace=True) # 안쓰는 열 삭제
df_train.drop(['percenta','percentb','percentTotal'],axis=1,inplace=True) # 안쓰는 열 삭제

df_train.columns
df_train.info()




# optimanl n_estimator
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()

print("optimal n_estimator is 30")


# optimal depths
 
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()

print("optimal depth is 8")
