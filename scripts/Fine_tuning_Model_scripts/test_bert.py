from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch

# Paths to the fine-tuned model and test dataset
model_path = "/content/drive/My Drive/fine_tuned_bert"

# Load the tokenizer and model
print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Load the test dataset
print("Loading test dataset...")
test_data = pd.read_csv("test_data.csv")
texts = test_data["text"].tolist()
true_labels = test_data["sentiment"].tolist()

# Tokenize the test data
print("Tokenizing test data...")
inputs = tokenizer(
    texts, return_tensors="pt", padding=True, truncation=True, max_length=128
)

data = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
dataloader = DataLoader(data, batch_size=16) 

# Perform inference in batches
print("Getting predictions...")
model.eval()
predicted_classes = []

with torch.no_grad():
    for batch in dataloader:
        batch_input_ids, batch_attention_mask = batch
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        # Perform inference
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
        predicted_classes.extend(batch_predictions)

# Compare predictions with true labels
print("Generating classification report...")
report = classification_report(true_labels, predicted_classes, zero_division=0)
print(report)

# Clear GPU memory to avoid fragmentation
torch.cuda.empty_cache()
