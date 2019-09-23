from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from TwitterSentimentAnalysis import sentiment_module as smd


api_key = "API_KEY"
api_secret = "API_SECRET_KEY"
access_token = "ACCESS_TOKEN"
access_token_secret = "ACCESS_TOKEN_SECRET"


class Listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value, confidence = smd.sentiment(tweet)
        print(tweet, '\n', sentiment_value, '\n', confidence)

        if confidence * 100 >= 80:
            output = open("twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track=["happy"])
