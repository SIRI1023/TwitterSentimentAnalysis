from pymongo import MongoClient
import pandas as pd
import random
from datetime import datetime, timedelta

# MongoDB connection URI
mongo_uri = "mongodb+srv://shri23govvala:cHLthAstbY6M2u@socialmediaanalysis.jfr0f.mongodb.net/?authSource=admin"
client = MongoClient(mongo_uri)

# Connect to the database and collection
db = client["SocialMediaAnalysis"]  
collection = db["TweetsWithSentiments"] 

# Load the CSV file
csv_file = "D:/TwitterSentimentAnalysis/scripts/data_csv_files/tweets_2021.csv" 
df = pd.read_csv(csv_file)

# Ensure columns 'text' and 'predicted_sentiment' exist in the dataframe
if "text" not in df.columns or "predicted_sentiment" not in df.columns:
    raise ValueError("CSV file must contain 'text' and 'predicted_sentiment' columns.")

# Generate random timestamps between February 2021 and July 2021
start_date = datetime(2021, 2, 1)
end_date = datetime(2021, 7, 31)

def random_date(start, end):
    """Generate a random datetime between start and end dates."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

# Prepare data for MongoDB
processed_tweets = []
for _, row in df.iterrows():
    tweet = {
        "id": int(row["id"]) if "id" in df.columns else None,  # Optional: add id if present
        "text": row["text"],
        "sentiment": row["predicted_sentiment"],
        "timestamp": random_date(start_date, end_date).strftime("%Y-%m-%d")
    }
    processed_tweets.append(tweet)

# Insert the processed tweets into MongoDB
result = collection.insert_many(processed_tweets)

# Output inserted IDs and verify insertion
print("Inserted tweet IDs:", result.inserted_ids)

# Verify insertion by fetching data from the collection
for tweet in collection.find().limit(5):  # Limit to first 5 for verification
    print(tweet)
