#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:11:10 2017

@author: paritosh
"""
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import re
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3) #3 is code ignore

corpus=[]
for i in range(0,len(dataset)):
    #cleaning the Texts Step2
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    #scaling lower case step 3
    review=review.lower()
    # remove non significent word(the is an etc like stopwords)step 4
    #nltk.data.path.append('/media/paritosh/Software/NLTK_Data')
    review=review.split() 
    ps=PorterStemmer() #Step 5 PorterStemmer like cleaning similer type of words loved to love
    review=[ps.stem(word) for word in review if not word in nltk.corpus.stopwords.words('english')]
    #step 6
    review=' '.join(review)
    corpus.append(review)

# create bag of word model spars matrix

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) #most relevent top 1500 
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



