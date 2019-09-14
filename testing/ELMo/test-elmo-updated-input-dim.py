import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import keras
import gensim
import re

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session() 
keras.backend.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

max_tweet_length = 50
vocab_size = 50000

# Fetch pre trained vecotrs
elmo_embed = hub.Module("./module_elmo2", trainable=False)

# Set up training data
trainData = pd.read_csv('../training-data.csv', sep=',', encoding='latin-1')
train_y = trainData['val'].values
train_x_raw = trainData['tweet'].values
train_x_unpadded = []
train_x_sizes = []
for tweet in train_x_raw:
    tweet_word_array = []
    for word in tweet.split(' '):
        word = re.sub(r"[^a-zA-Z0-9]+", '', word).lower()
        tweet_word_array.append(word)
        if len(tweet_word_array) < 50:
            train_x_sizes.append(50)
        else:
            train_x_sizes.append(len(tweet_word_array))
    train_x_unpadded.append(tweet_word_array)
train_x = pad_sequences(train_x_unpadded, maxlen=max_tweet_length, dtype=object, padding='pre', truncating='pre', value='')
    
    

# Set up test data
testData = pd.read_csv('../test-data.csv', sep=',', encoding='latin-1')
test_y = testData['val'].values
test_x_raw = testData['tweet'].values
test_x_unpadded = []
for tweet in test_x_raw:
    tweet_word_array = []
    for word in tweet.split(' '):
        word = re.sub(r"[^a-zA-Z0-9]+", '', word).lower()
        tweet_word_array.append(word)
    test_x_unpadded.append(tweet_word_array)
test_x = pad_sequences(test_x_unpadded, maxlen=max_tweet_length, dtype=object, padding='pre', truncating='pre', value='')

# Free up memory
del trainData
del train_x_raw
del train_x_unpadded
del testData
del test_x_raw
del test_x_unpadded

def ELMoEmbedding(x):
    return elmo_embed(
        inputs={
            'tokens': train_x,
            'sequence_len': train_x_sizes
        },
        signature="tokens",
        as_dict=True
    )["elmo"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
print(embedding.shape)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=[input_text], outputs=pred)
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(train_x_raw, train_y, epochs=50, batch_size=512)
result = model.evaluate(test_x_raw, test_y)
print("Accuracy: {0:.2%}".format(result[1]))
