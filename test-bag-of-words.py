import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

vocabulary_size = 5000
max_tweet_length = 50

# Set up training data
trainData = pd.read_csv('training-data.csv', sep=',', encoding='latin-1')
train_y = trainData['val'].values
train_x_raw = trainData['tweet'].values
train_x_one_hot = [one_hot(tweet, vocabulary_size, split=' ', lower=True, filters='!"£$%^&*()-_=+,<.>/?:;@#~[{]}\|`¬') for tweet in train_x_raw]
train_x = pad_sequences(train_x_one_hot, maxlen=max_tweet_length, padding='pre', truncating='pre')

# Set up test 
testData = pd.read_csv('test-data.csv', sep=',', encoding='latin-1')
test_y = testData['val'].values
test_x_raw = testData['tweet'].values
test_x_one_hot = [one_hot(tweet, vocabulary_size, split=' ', lower=True, filters='!"£$%^&*()-_=+,<.>/?:;@#~[{]}\|`¬') for tweet in test_x_raw]
test_x = pad_sequences(test_x_one_hot, maxlen=max_tweet_length, padding='pre', truncating='pre')

# Set up network
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=32, input_length=max_tweet_length, name='layer_embedding'))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train
model.fit(train_x, train_y, validation_split=0.05, epochs=50, batch_size=64)

# Test
result = model.evaluate(test_x, test_y)
print("Accuracy: {0:.2%}".format(result[1]))