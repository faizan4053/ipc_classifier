import pandas as pd
#import gensim
import matplotlib.pyplot as plt
import numpy as np
import nltk 
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
from sklearn.metrics import accuracy_score,confusion_matrix
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


tags=['A','B','C','D','E','F','G','H']


data=pd.read_csv('ipc_file1.tsv',sep='\t')

print(data.head(10))

#to count no of words in dataset
def count_words():
    print(data['text'].apply(lambda x: len(x.split(' '))).sum())


def plot_graph():
    plt.figure(figsize=(12,4))
    data.value.value_counts().plot(kind='bar')

#plot_graph()

def print_plot(index):
    d=data[data.index==index][['text','value']].values[0]
    if len(d)>0:
        print(d[0])
        print('value:',d[1])

#print_plot(10)
        

#no of words before cleaning
count_words()                  

#text cleaning
space=re.compile('[/(){}\[\]\|@,;]')
bad_symbol=re.compile('[^0-9a-zA-Z #+_]')
StopWords=set(stopwords.words('english'))    
#print(StopWords)       
#print(bad_symbol) 

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


def data_cleaner(dataset):
    dataset=BeautifulSoup(dataset,"lxml").text #HTML decoding
    dataset=dataset.lower()
    dataset=space.sub(' ',dataset)
    dataset=bad_symbol.sub('',dataset)
    dataset=' '.join(wordnet_lemmatizer.lemmatize(word) for word in dataset.split() if word not in StopWords)
    return dataset
    
data['text']=data['text'].apply(data_cleaner)

print_plot(10)

#no of words after cleaning
count_words()

#splitting data into training and validating dataset
x=data.text
y=data.value
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Naive-Bayes
#making a pipeline for converting dataset to a matrix of token counts
#then transform a count matrix to a normalized tfidf representation
#and the fitting that dataset 
processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',MultinomialNB())])
    

processor.fit(x_train,y_train)

from sklearn.metrics import classification_report 

y_predict=processor.predict(x_test) 

#print(y_predict)

print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
print(confusion_matrix(y_test,y_predict))

#Linear Support Vector Machine

from sklearn.linear_model import SGDClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('sgd',SGDClassifier(loss='hinge',
                penalty='l2',alpha=1e-3,random_state=42,max_iter=5,tol=None))])


processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)

print('accuracy %s' % accuracy_score(y_predict,y_test))
#print(classification_report(y_test,y_predict,target_names=tags))



#after changing some features

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('sgd',SGDClassifier(loss='squared_hinge',
                penalty='l2',alpha=1e-5,random_state=42,max_iter=5,tol=None))])


processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)

print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))









