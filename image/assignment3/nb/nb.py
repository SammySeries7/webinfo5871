# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 11:08:17 2023

@author: samsu
"""


import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.stem import SnowballStemmer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn import tree
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

df=pd.read_csv('C:\\Users\\samsu\\OneDrive\\Desktop\\info5871\\Dataset\\dataset for model\\data_total_50label_keepev_supervise.csv')
df=df[df['label'].notna()].copy()

lemma=list(df['lemma'])
stem=list(df['stemming'])

def stopword(string):
    added=['lol', 'mmm','nah','yay','uhh','lmao', 'lmfao','uhhh','yeah','ughhhh','uhhhh','umm','fyi','href','huh','oooffff']


# Initialize the stemmer
snow = SnowballStemmer('english')
def stemming(string):
    added=['lol', 'mmm','nah','yay','uhh','lmao', 'lmfao','uhhh','yeah','ughhhh','uhhhh','umm','fyi','href','huh','oooffff',
           'alon', 'alreadi', 'alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becaus', 'becom', 'befor', 'besid', 
           'cri', 'describ', 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'forti', 
           'henc', 'hereaft', 'herebi', 'howev', 'hundr', 'inde', 'mani', 'meanwhil', 'moreov', 'nobodi', 'noon', 'noth', 'nowher', 
           'onc', 'onli', 'otherwis', 'ourselv', 'perhap', 'pleas', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 
           'somewher', 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 'twelv', 'twenti', 'veri', 'whatev', 'whenc', 
           'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv']
    a=[snow.stem(i) for i in word_tokenize(string)]
    b=[i for i in a if i not in added]
    c=[i for i in b if (len(i)>=3)]
    return c


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['stemming'],df['label'],test_size=0.2, random_state=42)


Test_Y1=Test_Y
Encoder = LabelEncoder()

Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

'''
0=battery
1=cost
2=env
3=infrastructure
4=news
'''

# Lemmatization
'''
Tfidf_vect =TfidfVectorizer(input="content",  stop_words = "english", lowercase=True, max_df=0.9, min_df=2)

Tfidf_vect .fit(stem)  # create a sparse matrix

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
Train_X_Tfidf_dtm=Train_X_Tfidf.toarray()
Test_X_Tfidf_dtm=Test_X_Tfidf.toarray()
vocab = Tfidf_vect.get_feature_names_out()  # change to a list

MyDTM_tfidf_lemma_train=pd.DataFrame(Train_X_Tfidf_dtm,columns=vocab)
MyDTM_tfidf_lemma_test=pd.DataFrame(Test_X_Tfidf_dtm,columns=vocab)

'''
# binary cv
cv_bin =CountVectorizer(input="content",  stop_words = "english", lowercase=True, max_df=0.9, min_df=2, ngram_range=(1,2),
                                tokenizer=stemming, binary=True) #tokenizer=stemming)
cv_bin .fit(stem)  # create a sparse matrix

Train_X_cv = cv_bin.transform(Train_X)
Test_X_cv = cv_bin.transform(Test_X)
Train_X_cv_dtm=Train_X_cv.toarray()
Test_X_cv_dtm=Test_X_cv.toarray()
vocab = cv_bin.get_feature_names_out()  # change to a list

MyDTM_cv_lemma_train=pd.DataFrame(Train_X_cv_dtm,columns=vocab)
MyDTM_cv_lemma_test=pd.DataFrame(Test_X_cv_dtm,columns=vocab)


#lemma+stemming
#CountVectorizer, TfidfVectorizer
Tfidf_stemming =TfidfVectorizer(input="content",  stop_words = "english", lowercase=True, max_df=0.9, min_df=2, ngram_range=(1,2)) #,tokenizer=stemming
Tfidf_stemming .fit(lemma)  # create a sparse matrix

Train_X_Tfidf = Tfidf_stemming.transform(Train_X)
Test_X_Tfidf = Tfidf_stemming.transform(Test_X)
Train_X_Tfidf_dtm=Train_X_Tfidf.toarray()
Test_X_Tfidf_dtm=Test_X_Tfidf.toarray()
vocab = Tfidf_stemming.get_feature_names_out()  # change to a list

MyDTM_tfidf_lemma_train=pd.DataFrame(Train_X_Tfidf_dtm,columns=vocab)
MyDTM_tfidf_lemma_test=pd.DataFrame(Test_X_Tfidf_dtm,columns=vocab)


#normalize
MyDTM_normalize_lemma_train=MyDTM_tfidf_lemma_train.div(MyDTM_tfidf_lemma_train.sum(axis=1), axis=0).replace(np.nan,0)
MyDTM_normalize_lemma_test=MyDTM_tfidf_lemma_test.div(MyDTM_tfidf_lemma_test.sum(axis=1), axis=0).replace(np.nan,0)


#BernoulliNB
bn=BernoulliNB()
bn.fit(Train_X_cv_dtm, Train_Y)
predictions_bn=bn.predict(Test_X_cv_dtm)

# Use accuracy_score function to get the accuracy
print("GaussianNB Accuracy Score_linear -> ",accuracy_score(predictions_bn, Test_Y)*100)


print(confusion_matrix(predictions_bn, Test_Y))

'''
cv lemma+stemming
GaussianNB Accuracy Score_linear ->  64.1732283464567
[[23  2  0  5  1]
 [ 8 35  1  5 10]
 [18  9 45 17  8]
 [ 3  1  1 22  1]
 [ 0  1  0  0 38]]

cv stem
GaussianNB Accuracy Score_linear ->  63.38582677165354
[[20  1  0  5  1]
 [10 37  1  7 10]
 [18  7 45 16  8]
 [ 4  2  1 21  1]
 [ 0  1  0  0 38]]


cv lemma
GaussianNB Accuracy Score_linear ->  64.56692913385827
[[22  1  0  7  1]
 [ 8 36  1  4 11]
 [17  8 45 16  4]
 [ 5  2  1 22  3]
 [ 0  1  0  0 39]]


'''

#Gaussian
gb = GaussianNB()
gb.fit(Train_X_Tfidf_dtm, Train_Y)

predictions_gb=gb.predict(Test_X_Tfidf_dtm)


# Use accuracy_score function to get the accuracy
print("GaussianNB Accuracy Score_linear -> ",accuracy_score(predictions_gb, Test_Y)*100)


print(confusion_matrix(predictions_gb, Test_Y))

'''
lemma 4308 tfidf
GaussianNB Accuracy Score_linear ->  56.69291338582677
[[24  4  6 14  1]
 [ 4 23  5  7  3]
 [ 9  4 27  5  1]
 [11  7  8 21  4]
 [ 4 10  1  2 49]]

lemma+stem 4308 tfidf
GaussianNB Accuracy Score_linear ->  58.26771653543307
[[22  7  7 10  1]
 [ 6 25  4  6  4]
 [ 8  4 26  4  1]
 [12  4  8 27  4]
 [ 4  8  2  2 48]]

stem 4308 tfidf
GaussianNB Accuracy Score_linear ->  59.055118110236215
[[22  6  5 10  2]
 [ 6 26  5  6  2]
 [ 8  4 26  4  1]
 [13  6  9 27  4]
 [ 3  6  2  2 49]]

'''
#multinomial

nb = MultinomialNB()
nb.fit(Train_X_Tfidf_dtm, Train_Y)

predictions_nb=nb.predict(Test_X_Tfidf_dtm)


# Use accuracy_score function to get the accuracy
print("MultinomialNB Accuracy Score_linear -> ",accuracy_score(predictions_nb, Test_Y)*100)


print(confusion_matrix(predictions_nb, Test_Y))

def mapper(x):
    j={0:'battery',1:'cost',2:'env',3:'infrastructure',4:'news'}
    return(j[x])
    

predictions_nb_list=predictions_nb.tolist()
predictions_nb_result=list(map(mapper,predictions_nb_list))
Test_Y_list=list(Test_Y1)

lb=['battery','cost','env','infrastructure','news']
cm=confusion_matrix(predictions_nb_result, Test_Y_list,labels=lb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lb)
disp.plot()
plt.xticks(rotation=45)
plt.show()
f1_score(Test_Y, predictions_nb, average=None)


'''
lemma+stemming 4308 CV
MultinomialNB Accuracy Score_linear ->  70.86614173228347
[[28  4  1  6  1]
 [ 9 30  6  6  2]
 [ 7  3 36  3  0]
 [ 6  6  4 33  2]
 [ 2  5  0  1 53]]

lemma+stemming 4308 TFIDF
MultinomialNB Accuracy Score_linear ->  72.44094488188976
[[29  2  0  6  1]
 [ 9 30  5  6  1]
 [ 5  4 38  1  1]
 [ 6  6  4 35  3]
 [ 3  6  0  1 52]]


lemma+stemming 3000 TFIDF
MultinomialNB Accuracy Score_linear ->  72.04724409448819
[[30  2  1  7  2]
 [ 9 30  5  6  1]
 [ 5  4 37  1  1]
 [ 5  6  4 34  2]
 [ 3  6  0  1 52]]

lemma 4308 tfidf
MultinomialNB Accuracy Score_linear ->  72.44094488188976
[[29  3  0  6  0]
 [ 7 30  5  5  1]
 [ 6  3 37  3  0]
 [ 6  6  5 34  3]
 [ 4  6  0  1 54]]

stemming
MultinomialNB Accuracy Score_linear ->  74.01574803149606
[[30  2  0  5  1]
 [ 8 31  5  6  0]
 [ 6  3 38  1  1]
 [ 5  6  4 35  2]
 [ 3  6  0  2 54]]
'''

class_prob_sorted_0 = nb.feature_log_prob_[0, :].argsort()[::-1][:10]
class_prob_sorted_1 = nb.feature_log_prob_[1, :].argsort()[::-1][:10]
class_prob_sorted_2 = nb.feature_log_prob_[2, :].argsort()[::-1][:10]
class_prob_sorted_3 = nb.feature_log_prob_[3, :].argsort()[::-1][:10]
class_prob_sorted_4 = nb.feature_log_prob_[4, :].argsort()[::-1][:10]

print('Topic 0 :battery')
print(np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_0))

gr_bat=np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_0)
print('Topic 1 :cost')
print(np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_1))

gr_cost=np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_1)
print('Topic 2:env')
print(np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_2))

gr_env=np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_2)
print('Topic 3:Infrastructure')
print(np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_3))
gr_infra=np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_3)

print('Topic 4:news')
print(np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_4))
gr_news=np.take(Tfidf_stemming.get_feature_names(), class_prob_sorted_4)

'''
Topic 0 :battery
Topic 1:cost
Topic 2:env
Topic 3:Infrastructure
Topic 4:news
'''

import matplotlib.pyplot as plt


word_topic = np.vstack((gr_bat,gr_cost,gr_env,gr_infra,gr_news))
#print(word_topic)
word_topic=np.transpose(word_topic)
word_topic = word_topic.transpose()

num_top_words = 10
vocab_array = np.asarray(vocab)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10

num_topics=5
topics={0 :'battery',1:'cost',2:'env',3:'Infrastructure',4:'news'}
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('{}'.format(topics.get(t)))
    #top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    #top_words_idx = top_words_idx[:num_top_words]
    top_words = word_topic[t,:]
    #top_words_shares = word_topic[top_words_idx, t]
    for i, word in enumerate(top_words):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()