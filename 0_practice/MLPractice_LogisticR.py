
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

USAhousing = pd.read_csv('USA_Housing.csv')

USAhousing.info()
USAhousing.describe()
sns.pairplot(USAhousing) #모든 짝지을 수 있는 관계 파악
sns.displot(data=USAhousing, x='Price', kde=True)

USAhousing.corr()

sns.heatmap(USAhousing.corr()) #상관계수 표시

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

# print the intercept = B0
print(lm.intercept_)

# lm.coef_
# X.columns
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) # B1
coeff_df

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

## compare the actual price with the predicted one in the test set

y_test1=y_test.to_frame()
y_test1['Predictions']=predictions
y_test1.head()

sns.displot((y_test-predictions),bins=30)

# compute the erorrs and save them in the dataframe
y_test1['Error']=y_test1['Predictions']-y_test1['Price']
y_test1.head()

MSE=(sum(y_test1['Error']**2))/len(y_test1['Error'])
print(MSE)
     
MAE=(sum(y_test1['Error'].abs()))/len(y_test1['Error'])
print(MAE)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


