
# coding: utf-8

# In[108]:

import nltk 
from nltk.metrics import *
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from random import randint
from sklearn import svm 
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[ ]:

token_dict = []
train_data = []
train_labels = []
test_data = []
test_labels = []
import os 
path = os.getcwd() +"/ctgrs/"
for dirs,subdirs, files in os.walk(path):
    
    for file in files:
        
       
        file_path =  path + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        word_tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        lemmas = list(map(lambda w: lemmatizer.lemmatize(w), word_tokens))
        i=0
        for l in lemmas :
            i+=1
            if i%4!=0:
                train_data.append(l);
                train_labels.append(file);
            else:
                test_data.append(l);
                test_labels.append(file);
            
            
        
        
tfidf = TfidfVectorizer( stop_words='english')

tfs_train = tfidf.fit_transform(train_data)

tfs_test= tfidf.transform(test_data)


        
    
        


# In[118]:

classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(tfs_train, train_labels)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(tfs_test)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1


# In[119]:

classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(tfs_train, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(tfs_test)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1


# In[120]:

print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_labels, prediction_rbf))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_labels, prediction_linear))



