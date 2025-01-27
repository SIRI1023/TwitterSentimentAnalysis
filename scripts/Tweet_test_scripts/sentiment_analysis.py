# Sentiment Analysis Logic
# Sample Tweepy Script
import json
from random import choice

# Load mock tweets
with open("D:/SocialMediaSentimentAnalysis/data/mock_data.json", "r") as f:
    tweets = json.load(f)

# Sentiment analysis
sentiments = ["positive", "negative", "neutral"]
for tweet in tweets:
    tweet["sentiment"] = choice(sentiments)

# Save results
with open("D:/SocialMediaSentimentAnalysis/data/processed_data.json", "w") as f:
    json.dump(tweets, f, indent=4)

print("Sentiment analysis complete. Results saved to data/processed_data.json")
