import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Path to the fine-tuned BERT model and input/output files
model_path = "/content/drive/My Drive/fine_tuned_bert"
input_file = "/content/drive/My Drive/tweets_2021.csv"
output_file = "/content/drive/My Drive/tweets_with_sentiments.csv"

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the CSV file
df = pd.read_csv(input_file, encoding="latin1")
df["text"] = df["text"].str.replace(r"â€™", "'", regex=True)  # Fix single quotes
df["text"] = df["text"].str.replace(r"â€“", "-", regex=True)  # Fix dashes
df["text"] = df["text"].str.replace(r"[^\x00-\x7F]+", " ", regex=True)  # Remove non-ASCII
print(df.head())

# Make sure the "text" column exists in the CSV
if "text" not in df.columns:
    raise ValueError("The CSV file must contain a 'text' column.")

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Map numerical sentiment to textual labels
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

# Apply the model to each tweet
df["predicted_sentiment"] = df["text"].apply(lambda x: sentiment_map[predict_sentiment(x)])

# Save the results to a new CSV file
df.to_csv(output_file, index=False)

print(f"Sentiment analysis completed. Output saved to {output_file}.")
