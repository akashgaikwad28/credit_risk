import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()  



# Now, retrieve the variables from the environment
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "germen_credit_risk")
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "risk_check")
MONGO_DB_URL = os.getenv("MONGO_DB_URL")




MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder =  "artifacts"