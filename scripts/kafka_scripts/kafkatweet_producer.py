import json
from confluent_kafka import Producer
import tweepy

# Kafka configuration
producer = Producer({'bootstrap.servers': 'localhost:9092'})
topic_name = "tweet-stream"
max_tweets = 100  # Limit the number of tweets to collect

# Kafka delivery report callback
def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

# Tweepy API credentials (replace these with your own)
API_KEY = "WoSWZGfpFfJZCR3xcxww54eIX"
API_SECRET = "G7YgpoLUoPuT9Wox6DiaOGLAVp2ifhq7Z419LAqu0vV3U3wsNr"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAACNyyAEAAAAAiDsv3I80XoUVwB%2FHFiTouz3zVN0%3DzS4yrq31AxD3I30eZq1iZ4lO9ByM9bjw3EFKCNTTbksssXXyrT"

# Authenticate Tweepy client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Search tweets and send to Kafka
def fetch_tweets():
    tweet_count = 0
    query = "python OR kafka OR data lang:en"  # Define your search query

    try:
        print("Fetching tweets...")
        # Use search_recent_tweets to retrieve up to max_tweets
        response = client.search_recent_tweets(
            query=query,
            tweet_fields=["id", "text", "created_at"],
            max_results=90  # Maximum allowed per request
        )

        # Process and send tweets to Kafka
        for tweet in response.data:
            if tweet_count >= max_tweets:
                print("Max tweet limit reached.")
                break

            tweet_data = {
                "id": tweet.id,
                "text": tweet.text,
                "timestamp": str(tweet.created_at)
            }

            # Send to Kafka
            producer.produce(
                topic_name,
                key=str(tweet_data["id"]),
                value=json.dumps(tweet_data),
                callback=delivery_report
            )
            producer.flush()
            tweet_count += 1
            print(f"Tweet sent to Kafka: {tweet_data}")

    except Exception as e:
        print(f"Error fetching tweets: {e}")

# Run the tweet fetch function
if __name__ == "__main__":
    fetch_tweets()
    producer.flush()
    print("All tweets processed.")
