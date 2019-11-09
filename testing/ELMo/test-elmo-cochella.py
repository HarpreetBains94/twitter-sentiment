import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import keras
import re

from keras.models import Model
from tensorflow.python.keras.callbacks import Callback
from keras.layers import Input, Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

max_tweet_length = 15

def encode(le, labels):
    enc = le.transform(labels)
    return to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

def cleanText(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = re.sub(r'[,!@#$%^&*)(|/><";:.?\'\\}{]','',text)
    text = ' '.join(text.split(' ')[-15:])
    return text

# Fetch pre trained vecotrs
elmo_embed = hub.Module('./module_elmo2', trainable=False)

# Set up training data
train_data = pd.read_csv('../../datasets/Coachella-2015-2-DFE_good.csv', sep=',', encoding='latin-1')

y = train_data['coachella_sentiment'].values
lable_encoder = preprocessing.LabelEncoder()
lable_encoder.fit(y)
y_enc = encode(lable_encoder, y)

x = np.array(train_data['text'].apply(cleanText))

train_x, test_x, train_y, test_y = train_test_split(x, y_enc, test_size=0.1)

# Free up memory
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
def ELMoEmbedding(x):
    return elmo_embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype='string')
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(4, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#Train network
model.fit(
    train_x,
    train_y,
    epochs=10,
    batch_size=512,
    validation_split=0.1,
    callbacks=[EpochTestCallback((test_x, test_y))]
)
