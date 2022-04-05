import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

sns.histplot([0,1,2,3,4,5])
plt.show()

x = np.linspace(0,10,1000) #0부터 10까지 1000개를 뽑아주세요.
y = np.power(x,2) #x값을 거듭제곱합니다.
plt.plot(x,y)
plt.show()

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x,y)
plt.show()

slope, intercept, r, p, std_err = stats.linregress(x,y)
stats.linregress(x,y)
#intercept_stderr은 쓰려면 최신버전으로 업데이트 해야한다네?

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc,x))
mymodel


plt.plot(x,y)

plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.scatter(x,y)
#plt.show()

mymodel = np.poly1d(np.polyfit(x,y,3))
myline = np.linspace(1,22,100)
plt.plot(myline,mymodel(myline))
plt.show()

#-------------------------------------

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