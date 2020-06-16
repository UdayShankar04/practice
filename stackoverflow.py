# I will be using the Stack overflow dataset. I have grabbed around 2k sample
# for 4 tags iphone, java, javascript, python I will be building a deep
# learning model using keras.

# Import Libraries
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
plt.style.use('ggplot')
# %matplotlib inline
from IPython.core.display import display, HTML
display(HTML("<style>.container {width:100% !important;}</style>"))

# Read csv file from dir
data = pd.read_csv('stackoverflow.csv')
data.head()

# Class distributions
data.tags.value_counts()

import re
def decontracted(phrase):
  # specific
  phrase = re.sub(r"won't", "will not", phrase)
  phrase = re.sub(r"can\t", "can not", phrase)

  # general
  phrase = re.sub(r"n\t", "not", phrase)
  phrase = re.sub(r"\'re", "are", phrase)
  phrase = re.sub(r"\'s", "is", phrase)
  phrase = re.sub(r"\'d", "would", phrase)
  phrase = re.sub(r"\'ll", "will", phrase)
  phrase = re.sub(r"\'t", "not", phrase)
  phrase = re.sub(r"\'ve", "have", phrase)
  phrase = re.sub(r"\'m", "am", phrase)
  return phrase

#Stopwords
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

# Processing Text
from tqdm import tqdm
from bs4 import BeautifulSoup

preprocessed_posts =[]
#tqdm is for printing the status bar
for sentance in tqdm(data['post'].values):
  sentance = re.sub(r"http\S+", "", sentance)
  sentance = BeautifulSoup(sentance, 'lxml').get_text()
  sentance = decontracted(sentance)
  sentance = re.sub("\S*\d\S*", "",sentance).strip()
  sentance = re.sub('[^A-Za-z]+', ' ', sentance)
  
  sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

  preprocessed_posts.append(sentance.strip())
