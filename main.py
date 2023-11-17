import os
import pandas as pd #Reading in tabular data like csv files
import tensorflow as tf
import numpy as np # Used to wrap an extra dimension on data because model expects more than one batch

df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv', 'train.csv'))

#Printing out some data
# print(df.iloc[20]['comment_text'])
# print(df[df.columns[2:]].iloc[20])


X = df['comment_text']
Y = df[df.columns[2:]].values

# Max number of words stored in vocab
MAX_WORDS = 200000

#Converts words into integers (tokenize)
vectorizer = tf.keras.layers.TextVectorization(max_tokens=MAX_WORDS,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

