import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt

# MongoDB connection URI
mongo_uri = "mongodb+srv://shri23govvala:cHLthAstbY6M2u@socialmediaanalysis.jfr0f.mongodb.net/?authSource=admin"
client = MongoClient(mongo_uri)

# Connect to MongoDB database and collection
db = client["SocialMediaAnalysis"]
collection = db["TweetsWithSentiments"]

# Fetch data from MongoDB
data = list(collection.find({}, {"_id": 0}))  # Exclude _id field
df = pd.DataFrame(data)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Group by date and sentiment
sentiment_trends = df.groupby([df['timestamp'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
sentiment_trends.index = sentiment_trends.index.to_timestamp()

print(sentiment_trends)  # Check grouped data

# Plot sentiment trends over time
plt.figure(figsize=(12, 6))
for sentiment in sentiment_trends.columns:
    plt.plot(sentiment_trends.index, sentiment_trends[sentiment], label=sentiment.capitalize())

# Add chart labels and legend
plt.title("Sentiment Trends Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Tweets")
plt.legend(title="Sentiment")
plt.grid(True)

# Show the plot
plt.show()
