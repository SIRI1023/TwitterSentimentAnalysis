import json

# Mock tweets to simulate data ingestion
tweets = [
    {"id": 1, "text": "I love this products!", "timestamp": "2021-01-12"},
    {"id": 2, "text": "Terrible experience, never buying again.", "timestamp": "2021-02-08"},
    {"id": 3, "text": "Neutral tweet, no strong opinions.", "timestamp": "2021-03-17"},
    {"id": 4, "text": "Amazing customer service!", "timestamp": "2021-03-20"},
    {"id": 5, "text": "Product quality could be better.", "timestamp": "2021-04-05"}
]

# Save tweets to a JSON file in the data folder
with open("D:/SocialMediaSentimentAnalysis/data/mock_data.json", "w") as file:
    json.dump(tweets, file, indent=4)

print("Mock tweets saved to data/mock_data.json")
