import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense, CuDNNGRU
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

max_tweet_length = 15
vocab_size = 50000

# Fetch pre trained vecotrs
f = open('glove.42B.300d.txt','r')
model = {}
for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

# Set up training data
train_data = pd.read_csv('../training-data.csv', sep=',', encoding='latin-1')
train_y = train_data['val'].values
train_x_raw = train_data['tweet'].values
tokenizer = Tokenizer(filters='!"£$%^&*()-_=+,<.>/?:;@#~[{]}\|`¬')
tokenizer.fit_on_texts(train_x_raw)
word_index = tokenizer.word_index
word_index_length = len(word_index) + 1
train_x = pad_sequences(tokenizer.texts_to_sequences(train_x_raw), maxlen=max_tweet_length, padding='pre', truncating='pre')

# Set up test data
test_data = pd.read_csv('../test-data.csv', sep=',', encoding='latin-1')
test_y = test_data['val'].values
test_x_raw = test_data['tweet'].values
test_x = pad_sequences(tokenizer.texts_to_sequences(test_x_raw), maxlen=max_tweet_length, padding='pre', truncating='pre')

# Set up embedding matrix
embedding_matrix = np.zeros((word_index_length, 300))
for word, i in word_index.items():
    try:
        embedding_vector = model[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass
embedding_matrix = embedding_matrix[:vocab_size]

# Free up memory
del tokenizer
del word_index
del model
del train_data
del train_x_raw
del test_data
del test_x_raw

# Set up validation callback
class EpochTestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
# Set up network
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], input_length=max_tweet_length, trainable=False))
model.add(CuDNNGRU(units=256))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train
model.fit(
    train_x,
    train_y,
    validation_split=0.2,
    epochs=5,
    batch_size=512,
    callbacks=[EpochTestCallback((test_x, test_y))]
)