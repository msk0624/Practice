# Student name : 김혜성 
# ID number : 201821487 
# Email adress : ghtn2638@ajou.ac.kr 

import numpy as py
import pandas as pd 
import seaborn as sns 
import matplotlib
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
 
df = pd.read_csv("yelp.csv")
df.head()
df.info()


# 1. Use the “FacetGrid” function from the Seaborn library to create five histograms of text length of reviews with different numbers of stars: a histogram of 1-star reviews only, one of 2-star reviews only, one with 3-star reviews only, one with 4-star reviews only, and 5-star reviews only (i.e., histograms for the five number-of-star categories). Tip: sns.FacetGrid(yelp,col='stars')

df['len_text']  = df['text'].str.len()
facet = sns.FacetGrid(df, col='stars')
facet = facet.map(plt.hist, 'len_text')
plt.show()

# 2. Create a “boxplot” of text length for each number-of-star category. Tip: use the “sns.boxplot” function.

sns.boxplot(x = "stars", y = "len_text", data = df)
plt.show()

facet = sns.FacetGrid(df, col='stars')
facet = facet.map(sns.boxplot, 'len_text')
plt.show()

# 3. Create a “countplot” of the number of occurrences for each number-of-star category. Tip: use the “sns.countplot” function.
sns.countplot(df['stars'], color = '#682F2F')
plt.show()

# 4. Use the “groupby” function to get the mean values of the four numerical columns (i.e., cool, useful, funny, text length) within each number-of-star category.

df_2 = df.groupby('stars')
df_2.mean()

# 5. Use the “corr()” method to produce a correlation matrix of the four numerical columns.
df[['cool','useful','funny','len_text']].corr(method='pearson')


# 6. Do the NLP classification tasks

## 6-1. Create a dataframe called “yelp_class” that contains the columns of the “yelp” dataframe but for the 1- and 5-star reviews only, i.e., the 2-, 3-, and 4-star reviews excluded.

yelp_class = df[(df['stars'] == 1) | (df['stars'] == 5)]
yelp_class.reset_index(drop=True, inplace=True)
yelp_class['stars'].value_counts()
yelp_class.info()

# 6-2. Create two objects: X and y. X will be the “text” column of the “yelp_class” dataframe and y will be the “stars” column of the “yelp_class” dataframe.

X = yelp_class['text']
y = yelp_class['stars']


# 6-3. Import “CountVectorizer” and create a CountVectorizer object. Tip: import “CountVectorizer” from “sklearn.feature_extraction.text.”

import string
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

def text_process(mess):
    #"""
    #Takes in a string of text, then performs the following:
    #1. Remove all punctuation
    #2. Remove all stopwords
    #3. Returns a list of the cleaned text
    #"""
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer.fit(X)
print(len(bow_transformer.vocabulary_))

# 6-4. Use the “fit_transform” method on the CountVectorizer object and pass in X (the “text” column). Save this result by overwriting X. but i save this result in x.

print("I save this result in x. because it is comfort for me.")
x = bow_transformer.transform(X)
print(x)
x.shape

# 7. Use “train_test_split” to split up the data into “X_train,” “X_test,” “y_train,” and “y_test” with “test_size=0.3” and “random_state=101.”

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, 
                                                  y, 
                                                  test_size=0.30, random_state=101)

# 8. Import “MultinomialNB” and create an instance of the estimator. Call nb and fit nb using training data. Tip: import “MultinomialNB” from “sklearn.naive_bayes.”

from sklearn.naive_bayes import MultinomialNB
MNBmodel = MultinomialNB().fit(X_train, y_train)

# 9. Use the predict method of nb (i.e., the “MultinomialNB()” function) to predict whether the number of stars on a review is 5 or 1 from X_test. Tip: nb.predict(X_test)

MNB_pred = MNBmodel.predict(X_test)

# 10. Create a confusion matrix and classification report using these predictions and y_test. Tip: import the “confusion_matrix” and “classification_report” functions from “sklearn.metrics.”
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,MNB_pred))
print(classification_report(y_test,MNB_pred))

# 11. Print the most important five words in the prediction. Tip: use the “bow_transformer.get_feature_names()” function.
feature_to_coef = {
    word: coef for word, coef in zip(
        bow_transformer.get_feature_names(), MNBmodel.coef_[0]
    )
}

print("the 5 most important words for detecting stars:-------------")
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)