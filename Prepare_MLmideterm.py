## Student name : 김혜성 
## ID number : 201821487 
## Email adress : ghtn2638@ajou.ac.kr 


# 0. libarary 모두 불러오기
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


## Part A. Linear regression (Q1-Q5: 90 points)

# Q1. (15 points) Summarize the data set with descriptive statistics.
# Print the count, average, min, max statistics, and column names in the data set.
# Use the “head(),” “describe(),” “info()” functions in the Pandas library.
# 판다스의 기능을 이용해 데이터를 요약해라. 통계학 기술과 함께.

df = pd.read_csv('test_linear.csv')

print("Dataset have [] entires(This is [].), and they have some [] values in columns. ")
df.info()
df.head()
df.describe()
df.mean()
df.min()
df.max()
df.count()
df['Petrol_tax'].count() #특정 열도 가능


# Q2. (20 points) Visualize the data set. 
# First, generate a correlation graph. Use the “heatmap” function in the Seaborn library.
# Second, create a histogram of the dependent variable, i.e., consumption_of_petrol. Use the “histplot” function in the Seaborn library.

df.corr()
sns.heatmap(df.corr()) # 히트맵으로 상관계수 표시
sns.histplot(data=df, x=df['Petrol_Consumption'].dropna(),kde=True,color='darkred',bins=10)
plt.show()
# 특정 열 선택 후, bins는 그래프 몇개로 나누어표현, kde는 밀도표시


# Q3. (20 points) First, build a linear regression model. Use the “sklearn.linear_model” function. 
# Second, show the coefficient values of the four independent variables. Use the “coef_” attribute values of linear regressors.
# And use the “LinearRegression” function from the “sklearn” library)
# 여기서, Petrol_Consumption은 임의로 잡은 종속변수, 바꿔야 될수도 있음.

X = df.drop('Petrol_Consumption', axis=1)
y = df['Petrol_Consumption']

X_train, X_test, y_train, y_test = train_test_split(df.drop('Petrol_Consumption',axis=1), 
                                                    df['Petrol_Consumption'], test_size=0.40, 
                                                    random_state=101)

lnmodel = LinearRegression()
lnmodel.fit(X_train,y_train)

coeff_df = pd.DataFrame(lnmodel.coef_,X.columns,columns=['Coefficient'])
coeff_df

predictions = lnmodel.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()
print("This model accuracy is {}%".format(lnmodel.score(X_test, y_test)*100)) 

# Q4. (20 points) To evaluate the linear regression model, show the values of MAE (mean absolute error), MSE (mean squared error), and RMSE (root mean squared error). 
# Use the “metrics.mean_absolute_error” and “metrics.mean_squared_error” functions.
# MSE,MAE,RMSE는 예측값과 실제값의 차이들에 대한 제곱이나 절대값을 통해 음수를 처리한 뒤, 이들의 평균을 통해 구하는 것임. 그러므로 이를 줄이려면, 좀더 정확하게 선형모델을 만들어야 하는데, 이는 독립변수를 얼마나 잘 설정하느냐에 따라 달렸음. 좋은 선형모델 만들기는 설명변수를 얼마나 잘 설정하냐(필요한 개수만 적당히)넣는 것이 가장 중요함. 만약 설명변수를 너무 적게 넣어버리면 편이가 발생할수도 있고, 너무 많이 넣으면 편이는 발생하지 않으나, 자유도가 떨어지므로 효율적인 모델이라고 할 수 없게된다.

y_test1= y_test.to_frame()
y_test1['Predictions'] = predictions
y_test1

sns.histplot((y_test-predictions),bins=15,kde=True)
plt.show()

y_test1['Error']=y_test1['Predictions']-y_test1['Petrol_Consumption']
y_test1.head()

MSE=(sum(y_test1['Error']**2))/len(y_test1['Error'])
print(MSE)
     
MAE=(sum(y_test1['Error'].abs()))/len(y_test1['Error'])
print(MAE)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Q5. (15 points) Print your own thoughts on minimizing the error rates (MAE, MSE, and RMSE) to improve the predictions.
print("MSE, MAE, and RMSE are calculated by averaging the squares or absolute values of the differences between the predicted and actual values after processing negative numbers. Therefore, to reduce this, we need to create a more accurate linear model, which depends on how well the independent variable is set. To make a good linear model, it is most important to include how well the explanatory variables are set (only the necessary number is appropriate). If it has few explanatory variables, bias may appear large. And if too many explanatory variables are added, bias will not occur, but the degree of freedom is reduced, it cannot be said to be an efficient model.")


## Part B. Logistic regression (Q6-Q11: 110 points)

# Q6. (15 points) Summarize the data set with descriptive statistics. Print the count, average, min, max statistics, and column names in the data set. Use the “head(),” “describe(),” “info()” functions in the Pandas library.

print("Dataset have [] entires(This is [].), and they have some [] values in columns. ")
df = pd.read_csv('text_logistic.csv')
df.info()
df.head()
df.describe()
df.mean()
df.min()
df.max()
df.count()
df['age'].count() #특정 열도 가능

# Q7. (20points) Deal with missing values to clean the data set by removing the observations that have “etc” in the “industry” variable. 
# Use the “where” function in the Numpy library.
# 이 테스트셋에선 education에서 unknown을 드랍하도록 하겠음.
df['education'].head()
df['education'].unique()
np.where(df['education']=='unknown')
df.drop(df.index[np.where(df['education']=='unknown')], inplace=True)
df.drop(df.index[np.where(df['education']=='illiterate')], inplace=True)

# Q8. (20 points) Generate dummies of the categorical variable, i.e., the “industry” variable. 
# Use the “get_dummies” function in the Pandas library.
df['education'].value_counts()
# university.degree is reference group. because it has many values.

df = pd.get_dummies(data=df, columns=['education'], dtype=float)
df.columns
df.drop('education_university.degree', axis=1, inplace=True)


# Q9. (20 points) Build a logistic regression model using the “sklearn.logistic_model” function.
df['housing'].unique()

## df.drop(df.index[np.where(df['housing']=='unknown')], inplace=True) # 이거해도 되고

housingdf = df[(df['housing'] == 'yes') | (df['housing'] == 'no')]
housingdf['housing'].unique()

housingdf['housing'] = pd.get_dummies(data=housingdf['housing'], drop_first=True)
housingdf.info()
housingdf.corr()

X_variables = ['cons_price_idx','emp_var_rate','nr_employed']
y_variable = ['housing']

X = housingdf[X_variables]
y = housingdf[y_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
probs = logmodel.predict_proba(X_test)


# Q10. (15 points) Show the accuracy and precision metrics on the predictions of the logistic regression model. 
# Use the “sklearn.metrics” function.

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
# 만약 모델의 정확도가 잘못되면, X_variables를 바꿔야함.

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

# Q11. (20 points) Print your own suggestions on how to improve the predictions with more accuracy and more precision.
print("The failure of this regression analysis seems to be the failure of the dependent variable setting. A high-accuracy model could not be created by selecting a variable that had little correlation with other variables as the dependent variable.")
