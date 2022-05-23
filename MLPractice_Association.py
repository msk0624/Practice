import numpy as py
import pandas as pd 
import seaborn as sns 
import matplotlib
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

basket = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
basket.head()
basket.info()

#Converting the data frame into a list of lists 
records = []
for i in range (0, 7501):
    records.append([str(basket.values[i,j]) for j in range(0, 20)]) # 0~20 가지 순서대로 고객의 데이터를 리스트화 시킴
    
TE = TransactionEncoder()
array = TE.fit(records).transform(records)
#building the data frame rows are logical and columns are the items have been purchased 
transf_df = pd.DataFrame(array, columns = TE.columns_)
transf_df.head()
transf_df.describe() # 고객 데이터 각각을 장바구니에 담은 것을 일정 순서대로, 알파벳순으로 False와 True로 표현함.

#drop nan column
basket_clean = transf_df.drop(['nan'], axis = 1) #아무도 안담은 열 삭제
basket_clean.head()

#most popular items
count = basket_clean.loc[:,:].sum()

pop_item = count.sort_values(0, ascending = False).head(10)
pop_item = pop_item.to_frame()
pop_item = pop_item.reset_index()
pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item # 얘가 카운트해서 상위10

print(count)

# pop 시각화
plt.rcParams['figure.figsize'] = (10, 6)
matplotlib.style.use('dark_background')
ax = pop_item.plot.barh(x = 'items', y = 'count')
plt.title('Most popular items')
plt.gca().invert_yaxis()
plt.show() # 가장 많이 나온 아이템 추출하는 거 같네

#I chose 0.04 minimum support
a_rules = apriori(basket_clean, min_support = 0.04, use_colnames = True)
a_rules

rules = association_rules(a_rules, metric = 'lift', min_threshold = 1)
rules