# Student name : 김혜성 
# ID number : 201821487 
# Email adress : ghtn2638@ajou.ac.kr 

# 0. libarary 
import re
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from paramiko import Agent
import seaborn as sns
from statistics import mean
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics




## linear Regression

df_train = pd.read_csv('pisa2009train.csv')
df_test = pd.read_csv('pisa2009test.csv')

# 1. Data preview
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
# Data visualize using seaborn library

# A1. Summarize the data sets with descriptive statistics and others: e.g., the number of all the students, averages of the variables, correlations between the variables, etc.
df_train.info()
print("Train Dataset have 3663 entires(this is the number of students.), and they have some null values in columns. But in readingScore, there is no null values. so i want to know this value's average first.")
np.mean(df_train['readingScore'])
df_train.describe()
print("And this is all describe.")

# A2. Clean the data sets by dealing with missing values.
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
# if we have not enough data values. Then we use 'df[].fillna(df[].mean()) to replace missing values.
# df = df.fillna(df.mean())
# Below coding is example.
# # df_train.dropna(axis=0, subset='raceeth, +add column', inplace=True) 
# # df_train.fillna(df_train.mean(),inplace=True)
# But if i select this option. there is so many object to datapreprcoess. cause some columns conisists of binary data set. so i choose dropna option.

df_train.info()
df_test.info()

print("If I drop null values. I have 2414 students. and i think this number is enough to train data set. so i delete all na values.")
df_train.corr()
#I think it is so hard to see all corr numbers. so i visualize this.
sns.heatmap(df_train.corr()) #상관계수 표시
plt.show()

# A3. Convert the categorical variables into their dummies.
df_train['raceeth'].unique()
df_train['raceeth'].head()
df_train['raceeth'].value_counts()

df_train = pd.get_dummies(data=df_train, columns=['raceeth'], dtype=float)
df_train.columns
df_train.drop('raceeth_White', axis=1,inplace=True)
# Referrence catagory is raceeth_white becasue it has most common race in dataset
df_train.info()

# A4. Build a linear regression model (call it “lmScore”) using the training set to predict “readingScore” (the dependent variable) with all the remaining variables as independent variables.

X = df_train.drop('readingScore', axis=1)
y = df_train['readingScore']

X_train, X_vld, y_train, y_vld = train_test_split(df_train.drop('readingScore',axis=1), 
                                                    df_train['readingScore'], test_size=0.20, 
                                                    random_state=101)

lmscore = LinearRegression()
lmscore.fit(X_train,y_train)
# If i training this lmscore model by using keras library with vld dataset, i think it will improve my lmscore model.

print(lmscore.intercept_)
# This is Beta zero


# A5. Print your own explanations on which independent variables are key explainers in predicting the dependent variable.
# lm.coef_
coeff_df = pd.DataFrame(lmscore.coef_,X.columns,columns=['Coefficient'])
coeff_df
# This is other Betas.
print("The higher the coefficient, the greater the influence on the dependent variable. So I think that important indepedent variables are grade, expectBachelors, read30minsAday, and raceeth_American Indian/Alaska Native, Black, Hispanic.")

predictions = lmscore.predict(X_vld)
plt.scatter(y_vld,predictions)
plt.show()
print("This model accuracy is {}%".format(lmscore.score(X_vld, y_vld)*100)) 
# It's Accuracy is so bad..., as i mentioned, it is necessary to doing training with keras..!
# but now i'm doing with test.csv without vld training.
df_test.info()

X_test = df_train.drop('readingScore',axis=1)
y_test = df_train['readingScore']

predictions = lmscore.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()
print("This model accuracy is {}%".format(lmscore.score(X_test, y_test)*100))

y_test1= y_test.to_frame()
y_test1['Predictions'] = predictions
y_test1

sns.displot((y_test-predictions),bins=30,kde=True)
plt.show()

# compute the erorrs and save them in the dataframe
y_test1['Error']=y_test1['Predictions']-y_test1['readingScore']
y_test1.head()

# A6. Show such model evaluation indices as MAE, MSE, and RMSE.
MSE=(sum(y_test1['Error']**2))/len(y_test1['Error'])
print(MSE)
     
MAE=(sum(y_test1['Error'].abs()))/len(y_test1['Error'])
print(MAE)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))





## Logistic regression

# B1. Deal with missing values and clean the data: e.g., remove whitespace at the end of the column names.
# remove meaningless columns and na data
df_train = pd.read_csv('ks-projects-201612.csv', encoding='cp1252')
df_train.info()
df_train.drop(['Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16'], axis=1, inplace=True)
df_train.dropna(inplace=True)

# sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# maybe run over code... it cause error..! becasue of many dataset..!

# delete whitespace in columns end.
cols = df_train.columns.tolist()
for entry in np.arange(0,len(cols),1):
    cols[entry] = cols[entry].rstrip()
df_train.columns = cols
df_train.columns

df_train[['goal', 'pledged', 'usd pledged', 'backers']] = df_train[['goal', 'pledged', 'usd pledged', 'backers']].apply(pd.to_numeric, errors='coerce')
# df_train.info()

#Removing any random unhelpful ones. 
df_train['state'].value_counts() < 100
acc_states = ['failed', 'successful', 'live', 'canceled', 'suspended'] 
df_train = df_train[df_train['state'].isin(acc_states)] #일치하는 요소만 골라내는 함수
df_train['state'].unique()

print("Up to this point, I followed the website as a reference for efficient coding. If this wasn't the case, I'm really sorry, Professor.")

# B2. Visualize data (using Seaborn and Matplotlib) to show the distributions of continuous variables and compositions of categorical variables based on your interests.

plt.show()
sns.countplot(x='main_category', data=df_train, palette='GnBu_d')
sns.countplot(x='state', data=df_train, palette='GnBu_d')
sns.countplot(x='country', data=df_train, palette='GnBu_d')
sns.countplot(x='currency', data=df_train, palette='GnBu_d')
sns.displot(data=df_train, x=df_train['goal'].dropna(),kde=True,color='darkred',bins=30)
sns.displot(data=df_train, x=df_train['backers'].dropna(),kde=True,color='darkred',bins=30)
# I think that datapreprocess in continuous value is necessary.

plt.figure(figsize=(20,10))
sns.countplot(x='main_category', data=df_train, hue='state')
sns.countplot(x='country', data=df_train, hue='state')
sns.countplot(x='currency', data=df_train, hue='state')


# B3. Generate dummies of the categorical variables.
df_train.info()

df_train['currency'].value_counts() < 6000
df_train['country'].value_counts() < 6000
df_train['currency'].unique()
df_train['country'].unique()
#I choose currency first.

acc_currency = ['GBP','CAD','EUR','AUD']
df_train = df_train[df_train['currency'].isin(acc_currency)] #일치하는 요소만 골라내는 함수
df_train['currency'].unique()
df_train=pd.get_dummies(data=df_train, columns=['currency'], dtype=float)
df_train.columns

#choose GBP as a reference catagory.
df_train.drop('currency_GBP', axis=1, inplace=True)
df_train.info()

# B4. Do data processing to create a DataFrame containing only successful or failed projects.
successfaildf = df_train[(df_train['state'] == 'successful') | (df_train['state'] == 'failed')]
successfaildf['state']

successfaildf['goal'][successfaildf['goal'] > 5000000]
successfaildf = successfaildf[successfaildf['goal'] < 5000000]

successfaildf['pledged'][successfaildf['pledged'] > 10000]
successfaildf = successfaildf[(successfaildf['pledged'] < 10000) & (successfaildf['pledged'] > 0)]
successfaildf.info()
# Delete some highest values of pledged and goal..


# B5. Build a logistic regression model by splitting the data into train and test sets.
successfaildf['state'] = pd.get_dummies(data=successfaildf['state'], drop_first=True)
successfaildf.info()
successfaildf.corr()
#X_variables = []
#y_variable = []

X = successfaildf[['goal', 'backers', 'currency_AUD', 'currency_CAD', 'currency_EUR']]
# I want to add 'pledged' value. but some mystery reason make my logmodel acurracy 100... so i can't add 'pledged' value...
# I think maybe if i put pledge value in my model. Then it cause overfitting problem. so i delete this value.
y = successfaildf['state']
X.info()
y.info()


# B6. Show the accuracy and precision metrics on the predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
probs = logmodel.predict_proba(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probs[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# B7. Print your own suggestions on how to improve the predictions with more accuracy and precision.
print("Using Keras, it is expected that the accuracy will be higher if sufficient training is performed by splitting the train data into vld data before inserting the test data. Also, due to the problem of overfitting, the pledge variable could not be included here, but if this is also solved, it will be a good logistic model.")