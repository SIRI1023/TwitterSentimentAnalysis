from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import torch
import os

os.environ["WANDB_DISABLED"] = "true"

# Step 1: Load and preprocess data
print("Loading training and testing data...")
try:
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit()

# Debugging: Check initial data structure
print("\nInitial Train Data Sample:")
print(train_data.head())
print("Initial Test Data Sample:")
print(test_data.head())

# Ensure no missing or invalid rows
train_data = train_data.dropna(subset=["text", "sentiment"])
test_data = test_data.dropna(subset=["text", "sentiment"])

# Convert sentiment column to integer 
try:
    train_data["sentiment"] = train_data["sentiment"].astype(int)
    test_data["sentiment"] = test_data["sentiment"].astype(int)
except ValueError as e:
    print(f"Error converting sentiment column to integer: {e}")
    exit()

# Debugging: Check unique sentiment values
print("\nUnique sentiment values (train):", train_data["sentiment"].unique())
print("Unique sentiment values (test):", test_data["sentiment"].unique())
print(f"Number of rows in train_data after preprocessing: {len(train_data)}")
print(f"Number of rows in test_data after preprocessing: {len(test_data)}")

# Step 2: Define custom PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["sentiment"]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Return tokenized input and label
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Step 3: Initialize tokenizer and datasets
print("\nLoading tokenizer...")
model_path = "./bert-base-uncased"  
if not os.path.exists(model_path):
    print(f"Model directory '{model_path}' not found. Ensure the model files are available.")
    exit()

tokenizer = BertTokenizer.from_pretrained(model_path)

print("\nInitializing PyTorch datasets...")
train_dataset = SentimentDataset(train_data, tokenizer)
test_dataset = SentimentDataset(test_data, tokenizer)

# Debugging: Check dataset sizes
print(f"\nNumber of samples in train_dataset: {len(train_dataset)}")
print(f"Number of samples in test_dataset: {len(test_dataset)}")

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# Step 5: Load pre-trained model
print("\nLoading pre-trained BERT model...")
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

# Step 6: Initialize Hugging Face Trainer
print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Step 7: Train the model
print("\nStarting model training...")
try:
    trainer.train()
    print("Training complete.")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# Step 8: Save the fine-tuned model
print("\nSaving fine-tuned model...")
try:
    save_path = "/content/drive/My Drive/Colab Notebooks/sentimentAnalysis/"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Fine-tuning complete. Model saved to {save_path}")
except Exception as e:
    print(f"Error saving the fine-tuned model: {e}")
