# 0. libarary 

from statistics import mean
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from paramiko import Agent
import seaborn as sns
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

# 널값 처리 해야함.

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

print(lmscore.intercept_)
# This is Beta zero


# A5. Print your own explanations on which independent variables are key explainers in predicting the dependent variable.
# lm.coef_
coeff_df = pd.DataFrame(lmscore.coef_,X.columns,columns=['Coefficient'])
coeff_df
# This is other Betas.
print("The important indepedent variables are ")

predictions = lmscore.predict(X_vld)
plt.scatter(y_vld,predictions)
plt.show()
print("This model accuracy is {}%".format(lmscore.score(X_vld, y_vld)*100)) 
# It's Accuracy is so bad...

# Now doing with test.csv
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

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# delete whitespace in columns end.
cols = df_train.columns.tolist()
for entry in np.arange(0,len(cols),1):
    cols[entry] = cols[entry].rstrip()
df_train.columns = cols
df_train.columns

#Removing any random unhelpful ones. 
df_train['state'].value_counts() < 100
acc_states = ['failed', 'successful', 'live', 'undefined', 'suspended']
df_train = df_train[df_train['state'].isin(acc_states)]

# B2. Visualize data (using Seaborn and Matplotlib) to show the distributions of continuous variables and compositions of categorical variables based on your interests.

df_train[['goal', 'pledged', 'usd pledged', 'backers']] = df_train[['goal', 'pledged', 'usd pledged', 'backers']].apply(pd.to_numeric, errors='coerce')
df_train.info()

# I interested in goal. because it means that The funding goal is the amount of money that a creator needs to complete their project."

plt.show()

sns.countplot(x='main_category', data=df_train, palette='RdBu_r')

sns.countplot(x='Survived', hue='Sex', data=train)

sns.displot(data=train, x=train['Age'].dropna(),kde=False,color='darkred',bins=30)

sns.countplot(x='SibSp',data=train)

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')


