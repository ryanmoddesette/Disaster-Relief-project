# # Disaster Relief Project 
# The data set for this project consists of tweets from people experiencing natural disasters around the world. The tweets are classified by category some are further classified by having some sort of need. 
# Originally created as a jupyter notebook on Deepnote

#Modules used for project
# useful for opening files
import gdown
import zipfile

import os # accessing parts of your operating system
import re
import sys

# data visualization + manipulation
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


# Load the data.
disaster_tweets = pd.read_csv('disaster_data.csv',encoding ="ISO-8859-1")


# This function prints out a table containing all the tweets, along with their category labels
disaster_tweets.head()




#These were graphs and charts that were utilized on Deepnote to obetter visualize the dataset
_deepnote_run_altair(disaster_tweets, """{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","mark":{"type":"bar","tooltip":{"content":"data"}},"height":220,"autosize":{"type":"fit"},"data":{"name":"placeholder"},"encoding":{"x":{"field":"category","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}},"y":{"field":"need_or_resource","type":"nominal","sort":null,"scale":{"type":"linear","zero":true}},"color":{"field":"COUNT(*)","type":"quantitative","sort":null,"aggregate":"count","scale":{"type":"linear","zero":false}}}}""")
_deepnote_run_altair(disaster_tweets, """{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","mark":{"type":"bar","tooltip":{"content":"data"}},"height":220,"autosize":{"type":"fit"},"data":{"name":"placeholder"},"encoding":{"x":{"field":"tweet_id","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}},"y":{"field":"text","type":"nominal","sort":null,"scale":{"type":"linear","zero":true}},"color":{"field":"","type":"nominal","sort":null,"scale":{"type":"linear","zero":false}}}}""")


# # Pre-processing
needed_columns = ['text','category','need_or_resource'] #looking at the text of the tweet, the category, and whether it is a need or a resource
disaster_tweets_final = disaster_tweets[needed_columns]





disaster_tweets_final.head() #prints out first 5 entries in the newly reduced data table


category = 'Energy'
need_or_resource = 'need'

#prints the first 20 tweets related to energy problems
for t in disaster_tweets_final[disaster_tweets_final['category'] == category]['text'].head(20).values:
    print (t) 
    print('\n')
    


#Creates a wordcloud to visualize which words are most common in the tweets
category =  'Food'#@param {type:"integer"}
this_category_text = ''
this_category_text += t + ' '
    
wordcloud = WordCloud()   
wordcloud.generate_from_text(this_category_text)
plt.figure(figsize=(14,7))
plt.imshow(wordcloud, interpolation='bilinear')



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


def tokenize_vecs(text):
    clean_tokens = []
    for token in text_to_nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct): 
          # -PRON- is a special all inclusive "lemma" spaCy uses for any pronoun, we want to exclude these 
            clean_tokens.append(token)
    return np.array(clean_tokens)



X_text = np.array([process_lang_data(tweet) for tweet in disaster_tweets.text]) #X_test contains the tokenized tweets

print(X_text)

tweets = disaster_tweets.text
tweets = tweets.apply(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ',x))
# for text in disaster_tweets_final['text']:
#     for word in text:
#         lemmatizer.lemmatize(word)
lemmatizer = WordNetLemmatizer()
tweets = [lemmatizer.lemmatize(tweet) for tweet in tweets]
eng_stopwords = set(stopwords.words('english'))




X_text = disaster_tweets_final['text']
y = disaster_tweets_final['category']
vectorizer = CountVectorizer()
vectorizer.fit(X_text)




len_list = [len(tweet) for tweet in disaster_tweets.text]
plt.hist(len_list)
plt.hist(len_list)
plt.title('Distribution of Lengths of tweets') #plot of the length of the tweets
plt.xlabel('Length')
plt.ylabel('Number of Tweets')




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


X = vectorizer.fit_transform(tweets) #converting words to vectors


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



X_text



lm = LogisticRegression() #using linear regressoin
lm.fit(X_train, y_train) 



y_pred = lm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)


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




# Plot confusion matrix to analyze results
plot_confusion_matrix(y_test,y_pred)

