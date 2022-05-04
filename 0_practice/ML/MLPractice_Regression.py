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
