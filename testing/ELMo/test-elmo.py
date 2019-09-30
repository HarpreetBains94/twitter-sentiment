import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
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

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session() 
keras.backend.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

max_tweet_length=15

def cleanText(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = re.sub(r'[,!@#$%^&*)(|/><";:.?\'\\}{]','',text)
    text = ' '.join(text.split(' ')[-15:])
    return text

# Fetch pre trained vecotrs
elmo_embed = hub.Module('./module_elmo2', trainable=False)

# Set up training data
train_data = pd.read_csv('../training-data.csv', sep=',', encoding='latin-1')
train_y = np.array(train_data['val'])
train_x = np.array(train_data['tweet'].apply(cleanText))

# Set up test data
test_data = pd.read_csv('../test-data.csv', sep=',', encoding='latin-1')
test_y = np.array(test_data['val'])
test_x = np.array(test_data['tweet'].apply(cleanText))

# Free up memory
del train_data
del test_data

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
pred = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=[input_text], outputs=pred)
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#Train network
model.fit(
    train_x,
    train_y,
    epochs=5,
    validation_split=0.2,
    batch_size=512,
    callbacks=[EpochTestCallback((test_x, test_y))]
)
