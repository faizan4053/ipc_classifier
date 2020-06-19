import pandas as pd
#import gensim
import matplotlib.pyplot as plt
import numpy as np
import nltk 
from nltk.corpus import stopwords
from pprint import pprint
from time import time
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
from sklearn.metrics import accuracy_score,confusion_matrix
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report 





tags=['A','B','C','D','E','F','G','H']


data=pd.read_csv('ipc_file1.tsv',sep='\t')

print(data.head(10))

#to count no of words in dataset
def count_words():
    print(data['text'].apply(lambda x: len(x.split(' '))).sum())


def plot_graph():
    plt.figure(figsize=(12,4))
    data.value.value_counts().plot(kind='bar')

plot_graph()

def print_plot(index):
    d=data[data.index==index][['text','value']].values[0]
    if len(d)>0:
        print(d[0])
        print('value:',d[1])

print_plot(10)
        

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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#Naive-Bayes
#making a pipeline for converting dataset to a matrix of token counts
#then transform a count matrix to a normalized tfidf representation
#and the fitting that dataset 
processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',MultinomialNB())])
    

processor.fit(x_train,y_train)


y_predict=processor.predict(x_test) 

#print(y_predict)

print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)

#Linear Support Vector Machine

from sklearn.linear_model import SGDClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('sgd',SGDClassifier(loss='hinge',
                penalty='l2',alpha=1e-3,random_state=42,max_iter=5,tol=None))])


processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)

print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)



#after changing loss to squared_hinge and max_iter=50 alpha=1e-5

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('sgd',SGDClassifier(loss='squared_hinge',
                penalty='l2',alpha=1e-5,random_state=42,max_iter=50,tol=None))])
    
    


processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)

print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#after changing loss to modified_huber and max_iter 80 ,shuffle=True

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('sgd',SGDClassifier(loss='modified_huber',
                penalty='l2',alpha=1e-5,random_state=42,max_iter=80,tol=None,shuffle=True))])
    
    


processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)

print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#after changing loss to log and max_iter 100 ,shuffle=True and alpha=1e-6

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('sgd',SGDClassifier(loss='modified_huber',
                penalty='none',alpha=1e-5,random_state=42,max_iter=150,tol=None,shuffle=True))])
    
    


processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)

print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#testing for multiple parameters using GridSearchCV

from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(shuffle=True)),
])
    

parameters = {
    'clf__loss':('squared_hinge','hinge','modified_huber','log'),
    'clf__alpha': (0.00001, 0.000001,0.0000001,0.00000001),
    'clf__penalty': ('l1','l2', 'elasticnet','none'),
    'clf__max_iter': (20,50,80,100),
}

grid = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid.fit(x_train,y_train)
print("done in %0.3fs" % (time() - t0))
print()
print("Best score: %0.3f" % grid.best_score_)
print("Best parameters set:")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))   

y_predict=grid.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)
    

#using random searchCV

from sklearn.model_selection import RandomizedSearchCV


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(shuffle=True)),
])
    

parameters = {
    'clf__loss':('squared_hinge','hinge','modified_huber','log'),
    'clf__alpha': (0.00001, 0.000001,0.0000001,0.00000001),
    'clf__penalty': ('l1','l2', 'elasticnet','none'),
    'clf__max_iter': (40,50,80,100),
}

rscv = RandomizedSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

print("Performing random  search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
rscv.fit(x_train,y_train)
print("done in %0.3fs" % (time() - t0))
print()
print("Best score: %0.3f" % rscv.best_score_)
print("Best parameters set:")
best_parameters = rscv.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name])) 

y_predict=rscv.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)

#logistic regression

from sklearn.linear_model import LogisticRegression

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',LogisticRegression(n_jobs=4,C=1e5))])
#lr=LogisticRegression(n_jobs=1,C=1e5)
processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#logistic regression with some changes

from sklearn.linear_model import LogisticRegression

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',LogisticRegression(solver='lbfgs',n_jobs=-1,C=1e4,tol=1e-3))])
#lr=LogisticRegression(n_jobs=1,C=1e5)
processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#K Nearest Neighbours 

from sklearn.neighbors import KNeighborsClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',KNeighborsClassifier())])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)

#K Nearest Neighbours 

from sklearn.neighbors import KNeighborsClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',KNeighborsClassifier(n_neighbors=8,n_jobs=-1))])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)
#K Nearest Neighbours 

from sklearn.neighbors import KNeighborsClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',KNeighborsClassifier(n_neighbors=4,n_jobs=-1,weights='distance'))])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#decesion tree

from sklearn.tree import DecisionTreeClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',DecisionTreeClassifier())])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#decesion tree with changed parameters and hyperparameters

from sklearn.tree import DecisionTreeClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',DecisionTreeClassifier(splitter="random"))])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)


#random forest
from sklearn.ensemble import RandomForestClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',RandomForestClassifier())])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)



#random forest
from sklearn.ensemble import RandomForestClassifier

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',RandomForestClassifier(n_estimators=200,n_jobs=-1))])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)




#support vector machine

from sklearn.svm import SVC

processor=Pipeline([('vect',CountVectorizer()),
                ('tranform',TfidfTransformer()),
                ('multi',SVC(kernel="linear",gamma="auto"))])

processor.fit(x_train,y_train)
y_predict=processor.predict(x_test)
print('accuracy %s' % accuracy_score(y_predict,y_test))
print(classification_report(y_test,y_predict,target_names=tags))
count_misclassified = (y_test != y_predict).sum()
count_classified=(y_test==y_predict).sum()
print("misclassified:",count_misclassified)
print("classified:",count_classified)




















