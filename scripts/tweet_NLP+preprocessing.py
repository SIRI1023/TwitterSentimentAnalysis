from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import Tokenizer
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("NLP Preprocessing with PySpark") \
    .getOrCreate()

# Load CSV file
file_path = "/content/drive/My Drive/Colab Notebooks/SparkOutputs/tweets_2021.csv" 
tweets_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Step 1: Lowercase the text and remove special characters
tweets_df = tweets_df.withColumn("cleaned_text", lower(col("text")))
tweets_df = tweets_df.withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r"[^a-zA-Z0-9\s]", ""))

# Step 2: Tokenization
tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokens")
tokenized_df = tokenizer.transform(tweets_df)

# Step 3: Stopword removal
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))
processed_df = tokenized_df.withColumn("filtered_tokens", remove_stopwords_udf(col("tokens")))

# Select and save processed data
output_df = processed_df.select("id", "text", "cleaned_text", "filtered_tokens")
output_df.show(truncate=False)  # Show processed data

# Save to CSV
output_file_path = "/content/drive/My Drive/Colab Notebooks/SparkOutputs/processed_tweets.csv"  
output_df.write.csv(output_file_path, header=True)

# Stop Spark session
spark.stop()
