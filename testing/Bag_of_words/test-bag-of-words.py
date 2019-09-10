import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

max_tweet_length = 50
vocab_size = 50000

# Set up training data
trainData = pd.read_csv('../training-data.csv', sep=',', encoding='latin-1')
train_y = trainData['val'].values
train_x_raw = trainData['tweet'].values
tokenizer = Tokenizer(filters='!"£$%^&*()-_=+,<.>/?:;@#~[{]}\|`¬')
tokenizer.fit_on_texts(train_x_raw)
train_x = pad_sequences(tokenizer.texts_to_sequences(train_x_raw), maxlen=max_tweet_length, padding='pre', truncating='pre')

# Set up test 
testData = pd.read_csv('../test-data.csv', sep=',', encoding='latin-1')
test_y = testData['val'].values
test_x_raw = testData['tweet'].values
test_x = pad_sequences(tokenizer.texts_to_sequences(test_x_raw), maxlen=max_tweet_length, padding='pre', truncating='pre')

# Free up memory
del tokenizer
del trainData
del train_x_raw
del testData
del test_x_raw

# Set up network
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=max_tweet_length))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train
model.fit(
    train_x,
    train_y,
    validation_split=0.05,
    epochs=50,
    batch_size=1024,
    callbacks=[EpochTestCallback((test_x, test_y))]
)

# Test
result = model.evaluate(test_x, test_y)
print("Accuracy: {0:.2%}".format(result[1]))