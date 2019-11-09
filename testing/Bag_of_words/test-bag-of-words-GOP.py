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
from tensorflow.python.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

max_tweet_length = 15
vocab_size = 50000

def encode(le, labels):
    enc = le.transform(labels)
    return to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

# Set up training data
train_data = pd.read_csv('../../datasets/GOP_REL_ONLY_good.csv', sep=',', encoding='latin-1')

y = train_data['sentiment'].values
lable_encoder = preprocessing.LabelEncoder()
lable_encoder.fit(y)
y_enc = encode(lable_encoder, y)

x_raw = train_data['text'].values
tokenizer = Tokenizer(filters='!"£$%^&*()-_=+,<.>/?:;@#~[{]}\|`¬')
tokenizer.fit_on_texts(x_raw)
x = pad_sequences(tokenizer.texts_to_sequences(x_raw), maxlen=max_tweet_length, padding='pre', truncating='pre')

train_x, test_x, train_y, test_y = train_test_split(x, y_enc, test_size=0.1)

# Free up memory
del tokenizer
del train_data

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
model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=max_tweet_length))
model.add(CuDNNGRU(units=256))
model.add(Dense(3, activation='softmax'))
optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train
model.fit(
    train_x,
    train_y,
    epochs=10,
    batch_size=512,
    validation_split=0.1,
    callbacks=[EpochTestCallback((test_x, test_y))]
)