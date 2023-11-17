import os
import pandas as pd #Reading in tabular data like csv files
import tensorflow as tf
import numpy as np # Used to wrap an extra dimension on data because model expects more than one batch
from matplotlib import pyplot as plt

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

#Create Dataset
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, Y))
dataset = dataset.cache() #Data
dataset = dataset.shuffle(160000) #Shuffle
dataset = dataset.batch(16) # Creates batches of 16 samples
dataset = dataset.prefetch(8) #helps prevents bottlenecks
dataset_iterator = tf.data.NumpyIterator(dataset)

#Partition data into train, validation, and evaluation data.
train = dataset.take(int(len(dataset)*0.7))
validation = dataset.skip(int(len(dataset)*0.7)).take(int(len(dataset)*0.2))
evaluation = dataset.skip(int(len(dataset)*0.9)).take(int(len(dataset)*0.1))

#Deep learning
model = tf.keras.models.Sequential()

#Creates trait table for each word
model.add(tf.keras.layers.Embedding(MAX_WORDS+1, 32))

#LTSM captures patterns and predicts next "word" in sequence. Bidrectional allows scanning of data from left and right
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh')))

#Feature Extractors
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

#Final layer
model.add(tf.keras.layers.Dense(6, activation='sigmoid'))

model.compile(loss="BinaryCrossentropy", optimizer='Adam')
model.summary()

#Train model
history = model.fit(train, epochs=10, validation_data=validation)

#Plot Progress
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

#Make predictions
# input_text = vectorizer("I despise adam and bruno. Their are horrible hispanics, and I hope they fail math class. I am going to hurt them")
# result = model.predict(np.expand_dims(input_text, 0))

#Evaluation metrics
precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()
accuracy_metric = tf.keras.metrics.Accuracy()




