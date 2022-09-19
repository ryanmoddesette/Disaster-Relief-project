#!/usr/bin/env python
# coding: utf-8

# # Disaster Relief Project 
# The data set for this project consists of tweets from people experiencing natural disasters around the world. The tweets are classified by category some are further classified by having some sort of need. 
# 

# In[ ]:


#!pip install gensim 3


# In[ ]:


#@title Starter libraries (double click to take a look) { display-mode: "form" }
# useful for opening files
import gdown
import zipfile

import os # accessing parts of your operating system
import re
import sys

# data visualization + manipulation -- we've seen these many times
import numpy as np
import pandas as pd


# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.preprocessing import PolynomialFeatures # for polynomial model
from sklearn.pipeline import Pipeline

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# more sklearn model making
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from collections import Counter

# NLP
import string
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords' ,quiet=True)
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer
#ntlk.download('wordnet')
nltk.download('omw-1.4')
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
import wordcloud

import tweepy

# NN models
import tensorflow as tf
import tensorflow_datasets as tfds


from tensorflow import keras
#import keras
from tensorflow.keras import layers , activations , models , preprocessing, utils


# sequence data
from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

# text
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer

# NN and CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.wrappers.scikit_learn import KerasClassifier
import keras.optimizers as optimizers
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, DenseNet121

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#@title Load your dataset { display-mode: "form" }
# Run this every time you open the spreadsheet



import gdown
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




# In[ ]:


#@title If the previous cell fails to load data, use this cell
import re
import gdown
import seaborn as sns
import pandas as pd
import numpy as np
from torchtext.vocab import GloVe
from sklearn.model_selection import train_test_split



from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import requests, io, zipfile
import wordcloud
from wordcloud import WordCloud


# In[ ]:


# Load the data.
disaster_tweets = pd.read_csv('disaster_data.csv',encoding ="ISO-8859-1")


# In[ ]:


# This function prints out a table containing all the tweets, along with their category labels
disaster_tweets.head()


# In[ ]:


_deepnote_run_altair(disaster_tweets, """{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","mark":{"type":"bar","tooltip":{"content":"data"}},"height":220,"autosize":{"type":"fit"},"data":{"name":"placeholder"},"encoding":{"x":{"field":"category","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}},"y":{"field":"need_or_resource","type":"nominal","sort":null,"scale":{"type":"linear","zero":true}},"color":{"field":"COUNT(*)","type":"quantitative","sort":null,"aggregate":"count","scale":{"type":"linear","zero":false}}}}""")


# In[ ]:





# In[ ]:


_deepnote_run_altair(disaster_tweets, """{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","mark":{"type":"bar","tooltip":{"content":"data"}},"height":220,"autosize":{"type":"fit"},"data":{"name":"placeholder"},"encoding":{"x":{"field":"tweet_id","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}},"y":{"field":"text","type":"nominal","sort":null,"scale":{"type":"linear","zero":true}},"color":{"field":"","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}}}}""")


# # Pre-processing

# In[ ]:





# In[ ]:


_deepnote_run_altair(disaster_tweets, """{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","mark":{"type":"area","tooltip":{"content":"data"}},"height":220,"autosize":{"type":"fit"},"data":{"name":"placeholder"},"encoding":{"x":{"field":"text","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}},"y":{"field":"text","type":"nominal","sort":null,"scale":{"type":"linear","zero":true}},"color":{"field":"","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}}}}""")


# In[ ]:


needed_columns = ['text','category','need_or_resource']
disaster_tweets_final = disaster_tweets[needed_columns]


# In[ ]:


disaster_tweets_final.head()


# In[ ]:


category = 'Energy'
need_or_resource = 'need'

for t in disaster_tweets_final[disaster_tweets_final['category'] == category]['text'].head(20).values:
    print (t) 
    print('\n')
    


# In[ ]:


category =  'Food'#@param {type:"integer"}
this_category_text = ''
this_category_text += t + ' '
    
wordcloud = WordCloud()   
wordcloud.generate_from_text(this_category_text)
plt.figure(figsize=(14,7))
plt.imshow(wordcloud, interpolation='bilinear')


# In[ ]:


def process_lang_data(text):
  '''
    For a given text, go through the process of tokenizing, removing stopwords, 
    stemming / lemmatization, and removing punctuation. Return the cleaned text.
  '''
  cleaned_text = []
  punctuation = string.punctuation
  our_stopwords = stopwords.words('english')
  lemmatizer = WordNetLemmatizer()

  for token in word_tokenize(text):
    if token not in punctuation and token not in our_stopwords:
      clipped_token = lemmatizer.lemmatize(token)
      cleaned_text.append(clipped_token)

  return cleaned_text


# In[ ]:


def tokenize_vecs(text):
    clean_tokens = []
    for token in text_to_nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct): 
          # -PRON- is a special all inclusive "lemma" spaCy uses for any pronoun, we want to exclude these 
            clean_tokens.append(token)
    return np.array(clean_tokens)


# In[ ]:


X_text = np.array([process_lang_data(tweet) for tweet in disaster_tweets.text])

print(X_text)

tweets = disaster_tweets.text
tweets = tweets.apply(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ',x))
# for text in disaster_tweets_final['text']:
#     for word in text:
#         lemmatizer.lemmatize(word)
lemmatizer = WordNetLemmatizer()
tweets = [lemmatizer.lemmatize(tweet) for tweet in tweets]
eng_stopwords = set(stopwords.words('english'))


# In[ ]:


'''import gensim
# have to pre-tokenize


# take a look at the documentation to see what these parameters are changing!
w2vec_model = gensim.models.Word2Vec(tokenize, min_count = 1, window = 5, sg = 1)
w2vec_model.train(tokenize, total_examples = len(X_text),epochs=20)
words = list(w2vec_model)
print(words)'''


# In[ ]:


X_text = disaster_tweets_final['text']
y = disaster_tweets_final['category']
vectorizer = CountVectorizer()
vectorizer.fit(X_text)


# '''
# from sklearn.feature_extraction.text import CountVectorizer
# bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X_text.values)
# bow_transformer.fit(X_train)                             # fitting to our training data
#neural network, regularizing regresssion model



# In[ ]:


len_list = [len(tweet) for tweet in disaster_tweets.text]
plt.hist(len_list)
plt.hist(len_list)
plt.title('Distribution of Lengths of tweets')
plt.xlabel('Length')
plt.ylabel('Number of Tweets')



# In[ ]:


from collections import Counter

stpwrds = stopwords.words('english')
punctuation = string.punctuation

# try changing to visualize more or less words
num_words = 15

# text cleaning maintaining all of our text as one string
text = " ".join(disaster_tweets_final['text'])
# text = text.lower() # try adding this back in and see what happens!
text = "".join(_ for _ in text if _ not in punctuation)
text = [t for t in text.split() if t not in stpwrds and not t.isdigit()]

# We can use Counter to find the most frequent words in all our titles!
words = [_[0] for _ in Counter(text).most_common(num_words)]
frequency = [_[1] for _ in Counter(text).most_common(num_words)]

# Making our plot look nice!
plt.figure(figsize=(8,6));
ax = sns.barplot(x=frequency, y=words)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Most Common words used in Tweets");
plt.xlabel("Frequency", fontsize=14);
plt.yticks(fontsize=14);
plt.xticks(fontsize=14);


# In[ ]:


X = vectorizer.fit_transform(tweets)



# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


X_text


# In[ ]:


lm = LogisticRegression()
lm.fit(X_train, y_train)


# In[ ]:


y_pred = lm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)


# In[ ]:


def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = ['Energy', 'Food', 'Medical', 'None', 'Water']
  df_cm = pd.DataFrame(cm,index =labels,columns = labels)
  fig = plt.figure()
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5,3.5,4.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  plt.close()
# Plot confusion matrix
plot_confusion_matrix(y_test,y_pred)


# In[ ]:


#### Alternative - Using SBERT sentence transformer model ####

get_ipython().system('pip install sentence_transformers')
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(disaster_tweets.text, disaster_tweets.category, test_size=0.2, random_state=1)

s_transformer_embedder = SentenceTransformer('all-MiniLM-L6-v2')

#Compute embedding for both lists
X_train_embed = s_transformer_embedder.encode(X_train.values)
X_test_embed = s_transformer_embedder.encode(X_test.values)

print('--------------')
print('We first check out the shape of the embeddings produced by the SBert model')
print(f'X_train sentence embeddings shape is {X_train_embed.shape}')
print(f'X_test sentence embeddings shape is {X_test_embed.shape}')

# Train logistic regression
log_model = LogisticRegression() # create a LogisticRegression Model
log_model.fit(X_train_embed, y_train)

# Predict the y_preds
y_pred = log_model.predict(X_test_embed)
accuracy = accuracy_score(y_pred,y_test)
print('-------- RESULTS ------------')
print('.................................')
print(f'Accuracy of Bert-based Sentence-transformer model is {accuracy}')


# In[ ]:


def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = ['Energy', 'Food', 'Medical', 'None', 'Water']
  df_cm = pd.DataFrame(cm,index =labels,columns = labels)
  fig = plt.figure()
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5,3.5,4.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  plt.close()


# In[ ]:


# Plot confusion matrix
plot_confusion_matrix(y_test,y_pred)


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=0bf186be-d478-4087-a1ae-036f51b0a46a' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
