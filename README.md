# Twitter Sentiment Analysis

## Project Overview
This project demonstrates a real-time sentiment analysis pipeline for social media data, focusing on brand and customer feedback. It combines real-time data ingestion with Tweepy, NLP for sentiment classification, and data visualization to track trends.

## Features
- Real-time data ingestion using Tweepy API.
- Sentiment classification using Hugging Face Transformers and NLTK.
- Data visualization with Matplotlib for trend analysis.

## Tools and Technologies
- **Programming Languages**: Python
- **APIs**: Tweepy (Twitter API)
- **NLP Libraries**: Hugging Face Transformers, NLTK
- **Visualization**: Matplotlib, Seaborn
- **Streaming Framework**: Apache Kafka 
- **Database**: MongoDB

## Folder Structure
- `data/`: Mock datasets for testing the pipeline.
- `scripts/`: Python scripts for data ingestion, NLP processing, and visualization.
   ├── `data_csv_files` : Raw and processed CSV files
   ├──  `Fine_tuning_Model_scripts` : Scripts for fine-tuning BERT model
   ├──  `kafka_scripts` : Kafka scripts for real-time data processing
   ├── `logs` : Log files for tracking progress
   ├── `model_results` : Output predictions and evaluation metrics
   ├── `models` : Fine-tuned BERT model and tokenizer
   └── `Testing_tweet_data` : Initial phase of testing tweet data functionality
- `visualizations/`: Generated graphs and visualizations for sentiment trends.
- `docs/`: Additional documentation and architecture diagrams.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/SIRI1023/SocialMediaSentimentAnalysis.git
   cd SocialMediaSentimentAnalysis
