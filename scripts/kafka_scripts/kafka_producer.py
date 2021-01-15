from confluent_kafka import Producer
import json
import time

# Configure Kafka producer
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Delivery report callback
def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

# Simulated tweet data
tweets = [
    {"id": 1, "text": "I love this product!", "timestamp": "2021-01-12"},
    {"id": 2, "text": "Terrible experience, never buying again.", "timestamp": "2021-02-08"},
    {"id": 3, "text": "Neutral tweet, no strong opinions.", "timestamp": "2021-03-17"},
    {"id": 4, "text": "Amazing customer service!", "timestamp": "2021-03-20"},
    {"id": 5, "text": "Product quality could be better.", "timestamp": "2021-04-05"}
]

# Send tweets to Kafka topic
topic_name = "tweet-stream"
for tweet in tweets:
    producer.produce(
        topic_name,
        key=str(tweet['id']),
        value=json.dumps(tweet),
        callback=delivery_report
    )
    producer.flush()  # Ensure the message is sent
    time.sleep(1)  # Simulate streaming delay

producer.flush()
