from confluent_kafka import Consumer, KafkaException
import json
import csv
import datetime

# Configure Kafka consumer
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'tweet-group',
    'auto.offset.reset': 'earliest'
})

# Subscribe to the topic
consumer.subscribe(['tweet-stream'])

# Define a unique CSV file path
csv_file_path = f"tweets_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

# Initialize the CSV file with headers
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text"])  # CSV headers

print("Listening for tweets and saving to CSV...")

try:
    while True:
        msg = consumer.poll(timeout=1.0)  # Poll messages
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaException._PARTITION_EOF:
                continue
            else:
                print(f"Consumer error: {msg.error()}")
                break

        # Deserialize and process the message
        tweet = json.loads(msg.value().decode('utf-8'))
        print(f"Tweet received: {tweet}")

        # Save the tweet data to CSV
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([tweet.get("id"), tweet.get("text")])

except KeyboardInterrupt:
    print("Consumer closed.")
finally:
    consumer.close()
