import json
from collections import Counter
import matplotlib.pyplot as plt

# Load processed data
with open("D:/SocialMediaSentimentAnalysis/data/processed_data.json", "r") as f:
    tweets = json.load(f)

# Count sentiments
sentiment_counts = Counter(tweet["sentiment"] for tweet in tweets)

# Plot results
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=["green", "red", "blue"])
plt.title("Sentiment Analysis Result")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("D:/SocialMediaSentimentAnalysis/visualizations/sentiment_trends.png")
plt.show()
