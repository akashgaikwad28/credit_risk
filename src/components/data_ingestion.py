import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    artifact_folder: str  # Path to the folder where data will be stored

    def __init__(self, artifact_folder: str):
        self.artifact_folder = artifact_folder


class DataIngestion:
    def __init__(self, artifact_folder: str):
        self.data_ingestion_config = DataIngestionConfig(artifact_folder=artifact_folder)
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name):
        """Fetch data from MongoDB and return it as a pandas DataFrame."""
        try:
            # Establish connection to MongoDB
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[db_name][collection_name]

            # Convert MongoDB collection to DataFrame
            df = pd.DataFrame(list(collection.find()))

            # Check if data is empty
            if df.empty:
                raise CustomException(f"The MongoDB collection {collection_name} is empty.", sys)

            # Drop '_id' column if it exists (not needed for the data)
            if "_id" in df.columns:
                df = df.drop(columns=['_id'], axis=1)

            # Replace 'na' values with NaN
            df.replace({"na": np.nan}, inplace=True)

            # Check if the target column exists (assuming "Good/Bad" is the target)
            if "Risk" not in df.columns:
                raise CustomException(f"Target column 'Risk' not found in the MongoDB data", sys)

            logging.info(f"Data from MongoDB has been successfully fetched. Columns: {df.columns}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store_file_path(self) -> pd.DataFrame:
        """Exports the data into a feature store (CSV file)."""
        try:
            logging.info(f"Exporting data from MongoDB")

            # Define the path to save the raw data
            raw_file_path = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path, exist_ok=True)

            # Export data from MongoDB and convert it to DataFrame
            sensor_data = self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME,
                db_name=MONGO_DATABASE_NAME
            )

            logging.info(f"Saving exported data to feature store file path: {raw_file_path}")

            # Define the path for the CSV file
            feature_store_file_path = os.path.join(raw_file_path, 'german_credit_risk.csv')

            # Save the data to a CSV file
            sensor_data.to_csv(feature_store_file_path, index=False)

            # Check if the file was saved successfully
            if not os.path.exists(feature_store_file_path):
                raise CustomException(f"Failed to save data to {feature_store_file_path}", sys)

            logging.info(f"Data saved to {feature_store_file_path}")
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Path:
        """Initiate the data ingestion process and return the feature store file path."""
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            # Export data into the feature store and get the file path
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info("Data successfully ingested from MongoDB.")

            # Return the path to the saved feature store file
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)
