import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import keras
import pickle
import twitter_credentials
from flask import Flask, request, jsonify
import atexit

import re
import tweepy

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

def pimpJuice():
    config = tf.ConfigProto() 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    network = Network()
    query = ''
    while (query != 'quit'):
        query = input('Input search term: ')
        if (query != 'quit'):
            results = network.clasifyTweets(query)
            negativity = 0
            neutrality = 0
            positivity = 0
            for x in range(network.max_tweets):
                
                print('Tweet: ' + str(results[0][x]))
                print('Prediction: ' + str(results[1][x]))
                if str(results[1][x]) == 'negative':
                    negativity = negativity + 1
                elif str(results[1][x]) == 'neutral':
                    neutrality = neutrality + 1
                elif str(results[1][x]) == 'positive':
                    positivity = positivity + 1
            print(str( (negativity * 100) / len(results[1]) ) + '% negaitve')
            print(str( (neutrality * 100) / len(results[1]) ) + '% neutral')
            print(str( (positivity * 100) / len(results[1]) ) + '% positive')
    
class Network:
    def __init__(self):
        self.setUpTwitterApi()
        self.setUpModel()
            
    def setUpTwitterApi(self):
        auth = tweepy.OAuthHandler(twitter_credentials.API_KEY, twitter_credentials.API_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(auth)
        self.max_tweets = 10
        
    def decode(self, outputs):
        decoded_outputs = []
        for output in outputs:
            index = output.argmax()
            if index == 0:
                decoded_outputs.append('negative')
            elif index == 1:
                decoded_outputs.append('neutral')
            elif index == 2:
                decoded_outputs.append('positive')
        return decoded_outputs
    
    def setUpModel(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        keras.backend.set_session(sess)
        sess.run(tf.tables_initializer())
        atexit.register(sess.close)
        
        elmo_embed = hub.Module('./elmo', trainable=False)
        
        def ELMoEmbedding(x):
            return elmo_embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
        
        input_text = Input(shape=(1,), dtype='string')
        embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
        dense = Dense(256, activation='relu')(embedding)
        pred = Dense(3, activation='softmax')(dense)
        self.new_model = Model(inputs=[input_text], outputs=pred)
        optimizer = Adam(lr=1e-3)
        self.new_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.new_model.load_weights('final_model_weights.h5')
        
    def clasifyTweets(self, query):
        tweets = self.getTweets(query)
        result = self.new_model.predict(tweets)
        return (self.full_tweets, self.decode(result))
        
    def getTweets(self, query):
        query = query + ' -filter:media -filter:retweets -filter:links lang:en'
        self.full_tweets = [tweet.text for tweet in tweepy.Cursor(self.api.search, q=query).items(self.max_tweets)]
        tweets = []
        for tweet in self.full_tweets:
            tweet = tweet.replace('\n', ' ')
            tweet = tweet.replace('\r', ' ')
            tweet = re.sub(r'[,!@#$%^&*)(|/><";:.?\'\\}{]','',tweet)
            tweet = ' '.join(tweet.split(' ')[-20:])
            tweets.append(tweet)
        return np.asarray(tweets)

app = Flask(__name__)

def clasifyTweets(query):
    keras.backend.clear_session()
    network = Network()
    results = network.clasifyTweets(str(query))
    return jsonify(tweets=results[0], clasifications=results[1]) 

@app.route('/api/classify', methods=['POST'])
def onRequest():
    query = request.get_json()['query']
    results = clasifyTweets(query)
    return results

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')