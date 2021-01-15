from confluent_kafka import Consumer, KafkaException
import json

# Configure Kafka consumer
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'tweet-group',
    'auto.offset.reset': 'earliest'
})

# Subscribe to the topic
consumer.subscribe(['tweet-stream'])

print("Listening for tweets...")
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

except KeyboardInterrupt:
    print("Consumer closed.")
finally:
    consumer.close()