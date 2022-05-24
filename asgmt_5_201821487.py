# Student name : 김혜성 
# ID number : 201821487 
# Email adress : ghtn2638@ajou.ac.kr 


# 0. libarary 
from difflib import ndiff
import re
from tkinter import Y
from matplotlib import colors
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from paramiko import Agent
from regex import D
import seaborn as sns
from statistics import mean
from sklearn import metrics
import warnings
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

# A. Clustering and Profiling
# The data is related to direct marketing campaigns of a Portuguese banking institution. Cluster customers based on their attributes. The data set (the “bankmarketing.csv” file) has the following variables:
# 1. age (numeric)
# 2. job: type of job (categorical: admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
# 3. marital: marital status (categorical: divorced, married, single, unknown; note: divorced includes divorced and widowed)
# 4. education (categorical: basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown)
# 5. default: has credit in default? (categorical: no, yes, unknown)
# 6. housing: has housing loan? (categorical: no, yes, unknown)
# 7. loan: has personal loan? (categorical: no, yes, unknown)
# 8. contact: contact communication type (categorical: cellular, telephone)
# 9. month: last contact month of year (categorical: jan, feb, mar,..., nov, dec)
# 10. day_of_week: last contact day of the week (categorical: mon, tue, wed, thu, fri)
# 11. poutcome: outcome of the previous marketing campaign (categorical: failure, nonexistent, success)

# A1. Summarize the data set with descriptive statistics. You should print a count, average, min, max value, columns’ names in dataset. Use the “head(),” “describe(),” “info()” functions in the Pandas library.
df = pd.read_csv("bankmarketing.csv")
df.head()
df.describe()
df.info()

# A2. Prepare the data to build a model. Use the “preprocessing.LabelEncoder()” function in the sklearn library
# Label Encoding the object dtypes.

#remove unknown values
df = df[df != "unknown"]
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

# Later, I I will only use these category variables(age, marital, housing, loan, contact, poutcome) as explanatory variables. This is because there are too many variables when other categorical variables are converted to dummy variables through one hot enconding.
# df2_backup = df
# df.drop(['job','education','default','month','day_of_week'], axis = 1, inplace= True)
# df.columns

# if i need to one-hot encoding
# marital: marital status (categorical: divorced, married, single, unknown; note: divorced includes divorced and widowed)
# poutcome: outcome of the previous marketing campaign (categorical: failure, nonexistent, success)
# df=pd.get_dummies(data=df, columns=['marital'], dtype=float)
# df.drop('marital_married', axis = 1, inplace= True)
# df=pd.get_dummies(data=df, columns=['poutcome'], dtype=float)
# df.drop('poutcome_nonexistent', axis = 1, inplace= True)

# Doing labelencoding
s = (df.dtypes == 'object')
object_cols = list(s[s].index)

LE=LabelEncoder()
for i in object_cols:
    df[i]=df[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")
df.info()
df.head()
df.columns

sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

#correlation matrix
corrmat= df.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)
plt.show()

df['y'].value_counts()
df['default'].value_counts()

print("I will drop y, default. because I don't know it's means. also I will drop emp.var.rate, euribor3m. because it has high corr with other columns.")

df.drop(['y','default','emp.var.rate','euribor3m'], axis = 1, inplace= True)


# A3. Build a hybrid model with the k-means clustering and the agglomerative clustering. Use the KMeans and Elbow methods.
# A4. Visualize the clusters in terms of key predictors.


# if i want to drop binary columns
# bool_cols = [col for col in df if 
#                df[col].dropna().value_counts().index.isin([0,1]).all()]
# ds = ds.drop(bool_cols, axis=1)

#Creating a copy of data
ds = df.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")

#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
scaled_ds.head()
scaled_ds.columns


#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T
PCA_ds

#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()

# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()

print("Optimal cluster number is 4.")

#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
df["Clusters"]= yhat_AC

#Plotting the clusters

fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap)
ax.set_title("The Plot Of The Clusters")
plt.show()


# B. Association Rules
#Like Netflix, a TV show recommender system can always be built with context-based filtering or collaborative filtering. With a data set (the “TV_Shows-Association_Rules_Learning.csv” file), association rules can be identified by association rules using the “apriori” algorithm. the dataset includes details of more than 9,000 viewers watching TV shows. Each row represents the TV shows a viewer has watched together.


# B1. Prepare the data to find out association rules and drop the missing values. You can use the “preprocessing.TransactionEncoder()” function in the sklearn library. Please be reminded of the Boolean expression.

netflix = pd.read_csv("TV_Shows-Association_Rules_Learning.csv", header = None)
netflix.head()
netflix.info()

records = []
for i in range (0, 9690):
    records.append([str(netflix.values[i,j]) for j in range(0, 20)]) # 0~20 가지 순서대로 고객의 데이터를 리스트화 시킴
    
TE = TransactionEncoder()
array = TE.fit(records).transform(records)
#building the data frame rows are logical and columns are the items have been purchased 
transf_df = pd.DataFrame(array, columns = TE.columns_)
transf_df.head()
transf_df.describe() 
netflix_clean = transf_df.drop(['nan'], axis = 1) #모두 False인 컬럼 삭제
netflix_clean.head()


# B2. Visualize the most popular 10 items in the data set.
#most popular items
count = netflix_clean.loc[:,:].sum()
pop_item = count.sort_values(0, ascending = False).head(10)
pop_item = pop_item.to_frame()
pop_item = pop_item.reset_index()
pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item # 얘가 카운트해서 상위10

plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('dark_background')
ax = pop_item.plot.barh(x = 'items', y = 'count')
plt.title('Most popular items')
plt.gca().invert_yaxis()
plt.show() 

# B3. Show a list of association rules. Use the “apriori” and “association_rules” functions.
a_rules = apriori(netflix_clean, min_support = 0.04, use_colnames = True)
a_rules

rules = association_rules(a_rules, metric = 'lift', min_threshold = 1)
rules

# B4. Recommend several item sets of the TV shows based on the support, confidence, and lift measures.
print("The higher the lift, confidence, and support values, the more likely consumers are to watch the TV program. If recommended based on this, the items are Sex Education, Atypical, Mr. Robot, Ozark The Blacklist.")


