import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import gensim

from tensorflow.python.keras.models import Sequential
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

# Fetch pre trained vecotrs
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Set up training data
trainData = pd.read_csv('training-data.csv', sep=',', encoding='latin-1')
train_y = trainData['val'].values
train_x_raw = trainData['tweet'].values
tokenizer = Tokenizer(filters='!"£$%^&*()-_=+,<.>/?:;@#~[{]}\|`¬')
tokenizer.fit_on_texts(train_x_raw)
word_index = tokenizer.word_index
word_index_length = len(word_index) + 1
train_x = pad_sequences(tokenizer.texts_to_sequences(train_x_raw), maxlen=max_tweet_length, padding='pre', truncating='pre')

# Set up test data
testData = pd.read_csv('test-data.csv', sep=',', encoding='latin-1')
test_y = testData['val'].values
test_x_raw = testData['tweet'].values
test_x = pad_sequences(tokenizer.texts_to_sequences(test_x_raw), maxlen=max_tweet_length, padding='pre', truncating='pre')

# Set up embedding matrix
embedding_matrix = np.zeros((word_index_length, 300))
for word, i in word_index.items():
    try:
        embedding_vector = word2vec_model.get_vector(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass
embedding_matrix = embedding_matrix[: vocab_size]

# Free up memory
del tokenizer
del word_index
del word2vec_model
del trainData
del train_x_raw
del testData
del test_x_raw

# Set up network
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], input_length=max_tweet_length, trainable=False))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train
model.fit(train_x, train_y, validation_split=0.05, epochs=50, batch_size=512)

# Test
result = model.evaluate(test_x, test_y)
print("Accuracy: {0:.2%}".format(result[1]))