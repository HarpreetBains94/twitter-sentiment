import os
import tensorflow as tf
import numpy as np
import keras
import pickle
import twitter_credentials

import re
import tweepy

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def main():
    config = tf.ConfigProto( device_count = {'GPU': 1} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    model = Model()
    query = input('Input search term: ')
    results = model.clasifyTweets(query)
    positivity = 0
    for x in range(model.max_tweets - 1):
        print('Tweet: ' + ' '.join(results[0][x]))
        print('Prediction: ' + str(results[1][x][0])) 
        if (round(results[1][x][0]) == 1.0):
            positivity = positivity + 1
    print(str( (positivity * 100) / len(results[1]) ) + '% positive')
    
class Model:
    def __init__(self):
        self.setUpTwitterApi()
        self.setUpModel()
            
    def setUpTwitterApi(self):
        auth = tweepy.OAuthHandler(twitter_credentials.API_KEY, twitter_credentials.API_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(auth)
        self.max_tweets = 100
    
    def setUpModel(self):
        self.new_model = load_model('glove_training_model.h5')
        self.max_tweet_length = 50
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
    def clasifyTweets(self, query):
        self.getTweets(query)
        x = pad_sequences(self.tokenizer.texts_to_sequences(self.tweets), maxlen=self.max_tweet_length, padding='pre', truncating='pre')
        result = self.new_model.predict(x)
        return (self.tweets, result)
        
    def getTweets(self, query):
        query = query + ' -filter:media -filter:retweets -filter:links lang:en'
        searched_tweets = [tweet.text for tweet in tweepy.Cursor(self.api.search, q=query).items(self.max_tweets)]
        tweets = []
        for searched_tweet in searched_tweets:
            tweet = []
            searched_tweet = searched_tweet.replace('\n', ' ')
            searched_tweet = searched_tweet.replace('\r', ' ')
            for word in searched_tweet.split(' '):
                tweet.append(re.sub(r"[^a-zA-Z0-9]+", '', word).lower())
            tweets.append(tweet)
        self.tweets = tweets

main()