# Student name : 김혜성 
# ID number : 201821487 
# Email adress : ghtn2638@ajou.ac.kr 


# 0. libarary 
import re
from tkinter import Y
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from paramiko import Agent
import seaborn as sns
from statistics import mean
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

## Decision Tree

df_train = pd.read_csv('titanic_cleaned.csv')

# 1. Summarize the data set with descriptive statistics. That is, print the count, average, min, max values, 
# and the columns’ names in the data set. You can use the “head(),” “describe(),” “info()” functions in Pandas library.

df_train.head()
df_train.info()
df_train.describe()

# 2. Split up the data into a training set and a test set. You can use the “train_test_split” function 
# in the “sklearn.model_selection” path. Your codes should be set the following values: i.e., test_size=0.30, 
# random_state=101. The “random_state” value is used for initializing the internal random number generator 
# which decides how to split all the data into the train and test sets.

X = df_train.drop('Survived', axis=1)
y = df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                  y, 
                                                  test_size=0.30, random_state=101)

X.head()
y.head()

# 3. Print the feature importance values to see which predictors are more important in predicting 
# the classification than the other predictors. Use the “feature_importances_” attribute.

# 4. Build a decision tree model and a random forest classifier model to predict the target variable, 
# i.e., “Survival.” You can use the “sklearn.tree” library path to develop the decision tree model 
# and the “sklearn.ensemble” path to develop the random forest model.


## decisiontree
clf = DecisionTreeClassifier(max_depth = 4)
clf.fit(X_train, y_train)

clf_pred = clf.predict(X_test)

print(confusion_matrix(y_test,clf_pred))
print(classification_report(y_test,clf_pred))

pd.DataFrame(clf.feature_importances_, X_train.columns, columns=['Feature Importance'])

## randomforest

rfc = RandomForestClassifier(max_depth = 4, n_estimators=20)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

pd.DataFrame(rfc.feature_importances_, X_train.columns, columns=['Feature Importance'])

# 5. Visualize the two prediction models, i.e., the decision trees of the two classification models, 
# as shown in the example below. You can use the “tree.plot_tree” function in the “sklearn” library.
# Also, the “plt.subplots” function in the Matplotlib library will be helpful for your coding.

tree.plot_tree(clf, feature_names=X.columns, filled=True)
plt.show()

tree.plot_tree(rfc.estimators_[0], feature_names=X.columns, filled=True)
plt.show()

# 6. Evaluate the alternatives models: i.e., logistic regression, random forest, kNN, and naive Bayes. That is, compare the four algorithms in terms of their prediction performance based on accuracy and precision indices. You can use the “classification_report” and “confusion_matrix” functions in “sklearn.metrics” library path.

models=[]
models.append(('LR',LogisticRegression()))
models.append(('RFC',RandomForestClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))

for name, model in models:
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))