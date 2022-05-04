# 0.
import pandas

mydataset = { 'cars' : ["BMW", "Volvo", "Ford"],
             'passings':[3, 7, 2]
             }

myvar = pandas.DataFrame(mydataset)
print(mydataset)
print(myvar)

# 1.
import pandas as pd
a = [1,7,2]
myvar = pd.Series(a)
print(myvar)

myvar = pd.Series(a, index = ["x","y","z"])
print(myvar)

# 2.
calories = {"day1": 420, "day2": 380, "day3":390}
myvar = pd.Series(calories)
print(myvar) 
## 얘는 day가 인덱스가 되네

myvar = pd.Series(calories , index=["day1","day2"])
## 일부만 선택

# 3.
data = {
    "calories" : [420,380,390],
    "duration" : [50, 40, 45]
}

df = pd.DataFrame(data)
print(df)
## 인덱스 기본값으로 하고싶으면

print(df.loc[0])
print(df.loc[[0,1]])
## 행추출인데 인덱스 말고 str로 넣어도 추출가능할걸

print(df.iloc[0])
print(df.iloc[[0,1]])
## 행추출인데 얘가 아마 인덱스 위주인가 그럼

print(df["calories"])
print(df["calories"].loc[0])
## 열추출

# 4.
import os
os.getcwd()
## 작업폴더 확인하기

df = pd.read_csv('data.csv')
print(df)
print(df.head(10))
print(df.info())
## info 확인해야 하는 이유가 널값 있는지 없는지 확인 위해서. 실제로 calories에 대해선 5개의 미싱 밸류가 있다.
## 그래서 이 널값을 채우든지, 이 행을 아예 그냥 삭제하든지 하는 전처리 과정이 필요하다!
## pd.options.display.max_rows 로 출력되는 최대개수 설정 가능

# 5.
import matplotlib.pyplot as plt
df.plot()
plt.show()

df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')
plt.show()