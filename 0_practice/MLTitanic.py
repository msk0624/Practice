import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from paramiko import Agent
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

sns.setstyle('whitegrid')
sns.countplot(x='Survived', data=train, palette='RdBu_r')

sns.countplot(x='Survived', hue='Sex', data=train)

sns.displot(data=train, x=train['Age'].dropna(),kde=False,color='darkred',bins=30)

sns.countplot(x='SibSp',data=train)

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

train=pd.get_dummies(data=train, columns=['Sex','Embarked'], drop_first=True)
train.head()

train_final=train.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
train_final.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_final.drop('Survived',axis=1), 
                                                    train_final['Survived'], test_size=0.30, 
                                                    random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
probs = logmodel.predict_proba(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import roc_curve, auc

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



#--------------------------밑에서 부터는 추가 학습-------------------------------------------

train.head()

import re

def found_title(x):
    return re.search(', (.+?). ', x).group(1)

train['Title']=train['Name'].apply(found_title)
train.head()

train['Title'].value_counts()

train=pd.get_dummies(data=train, columns=['Title']) 
train.columns
train.drop(['Title_Capt', 'Title_Col',
       'Title_Don', 'Title_Jonkheer', 'Title_Lady', 'Title_Major',
       'Title_Mlle', 'Title_Mme', 'Title_Ms', 'Title_Sir', 'Title_th'], axis=1, inplace=True)

train.drop(['Name','Ticket', 'PassengerId'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
probs = logmodel.predict_proba(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probs[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

X_trainval, X_test, y_trainval, y_test = train_test_split(train_final.drop('Survived',axis=1), 
                                                    train_final['Survived'], test_size=0.30, 
                                                    random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=101)

val_scores = []
c_n=np.logspace(-3,3,7)
for i in c_n:
    print(i)
    logmodel = LogisticRegression(C=i,solver='liblinear')
    logmodel.fit(X_train, y_train)
    val_scores.append(logmodel.score(X_val, y_val))
print("best validation score: {:.3f}".format(np.max(val_scores)))
best_c_n=c_n[np.argmax(val_scores)]
print("best c_n: {}".format(best_c_n))

logmodel=LogisticRegression(C=best_c_n,solver='liblinear')
logmodel.fit(X_trainval, y_trainval) # note that this time we fit the model with best hyperparameters on training and validation sets
print("test score: {:.3f}".format(logmodel.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(train_final.drop('Survived',axis=1), 
                                                    train_final['Survived'], test_size=0.30, 
                                                    random_state=101)
cross_val_scores=[]
for i in c_n:
    logmodel = LogisticRegression(C=i,solver='liblinear')
    scores=cross_val_score(logmodel, X_train, y_train, cv=10)
    cross_val_scores.append(np.mean(scores))
print("best cross-validation score: {:.3f}".format(np.max(cross_val_scores)))
best_c_n=c_n[np.argmax(cross_val_scores)]
print("best c_n: {}".format(best_c_n))

logmodel=LogisticRegression(C=best_c_n,solver='liblinear')
logmodel.fit(X_train, y_train)
print("test score: {:.3f}".format(logmodel.score(X_test, y_test)))


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

logmodel = LogisticRegression(solver='liblinear')

# A parameter grid for logistic regression
params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }

grid = GridSearchCV(estimator=logmodel, param_grid=params, cv=10)

grid.fit(X_train, y_train)

print('\n Best estimator:')
print(grid.best_estimator_)
print('\n Best score:')
print(grid.best_score_)
print('\n Best parameters:')
print(grid.best_params_)

predictions = grid.best_estimator_.predict(X_test)
ac = accuracy_score(y_test, predictions)
print("accuracy: %f" % (ac))