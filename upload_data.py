from pymongo.mongo_client import MongoClient
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Step 1: Load environment variables from the .env file
load_dotenv()

# Retrieve MongoDB URL, Database Name, and Collection Name from environment variables
uri = os.getenv("MONGO_DB_URL")
DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# Step 2: Ensure the MongoDB URI is available
if uri is None:
    raise ValueError("MONGO_DB_URL environment variable is not set.")
if DATABASE_NAME is None:
    raise ValueError("MONGO_DATABASE_NAME environment variable is not set.")
if COLLECTION_NAME is None:
    raise ValueError("MONGO_COLLECTION_NAME environment variable is not set.")

# Step 3: Establish connection with MongoDB
client = MongoClient(uri)

# Step 4: Path to the CSV file you want to upload
csv_file_path = r"C:\Users\akash\Credit_Risk\notebooks\german_credit_data.csv"  # Update the path if necessary

# Step 5: Read CSV file into a Pandas DataFrame
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Step 6: Check and drop any unnamed columns (typically, index columns in CSV files)
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Step 7: Convert the DataFrame to JSON format that MongoDB accepts (list of dictionaries)
json_record = df.to_dict(orient="records")

# Step 8: Insert the records into MongoDB collection
try:
    # Insert data into the specified collection
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    print("Data inserted successfully.")
except Exception as e:
    print(f"Error inserting data into MongoDB: {e}")
