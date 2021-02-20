import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Download the dataset
api = KaggleApi()
api.authenticate()

dataset = "jp797498e/twitter-entity-sentiment-analysis"
output_dir = "data/"
os.makedirs(output_dir, exist_ok=True)
api.dataset_download_files(dataset, path=output_dir, unzip=True)
print(f"Dataset downloaded to {output_dir}")

# Step 2: Load the dataset
csv_file_path = f"{output_dir}twitter_training.csv"
column_headers = ["id", "category", "sentiment", "text"]
data = pd.read_csv(csv_file_path, names=column_headers, header=None)
print("Dataset loaded successfully!")
print(f"Columns in dataset: {data.columns}")

# Step 3: Inspect sentiment values
print(f"Unique sentiment values before mapping: {data['sentiment'].unique()}")

# Step 4: Map sentiments to numerical values

sentiment_mapping = {"positive": 2, "neutral": 1, "negative": 0}
data["sentiment"] = data["sentiment"].str.lower()  # Convert to lowercase for consistency
data["sentiment"] = data["sentiment"].map(sentiment_mapping)

# Handle unmapped sentiments
unmapped_rows = data["sentiment"].isnull().sum()
if unmapped_rows > 0:
    print(f"Unmapped sentiment values found: {unmapped_rows}. Dropping them...")
    data = data.dropna(subset=["sentiment"])

# Ensure there are rows remaining
if data.shape[0] == 0:
    print("No valid rows remain in the dataset after filtering. Exiting...")
    exit()

# Ensure sentiment values are integers
data["sentiment"] = data["sentiment"].astype(int)

# Step 5: Preprocess the dataset
data = data[["text", "sentiment"]]
data.dropna(inplace=True)

# Log dataset size
print(f"Dataset size after preprocessing: {data.shape}")
print(f"Unique sentiment values: {data['sentiment'].unique()}")

# Step 6: Split the dataset into training and testing sets
try:
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train data size: {train_data.shape}")
    print(f"Test data size: {test_data.shape}")
except ValueError as e:
    print(f"Error splitting the dataset: {e}")
    exit()

# Step 7: Save the processed data
train_csv_path = f"{output_dir}train_data.csv"
test_csv_path = f"{output_dir}test_data.csv"

train_data.to_csv(train_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

print(f"Data preprocessing complete. Train dataset saved to {train_csv_path}, test dataset saved to {test_csv_path}.")
