from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

# Define target column globally
TARGET_COLUMN = "Risk"

@dataclass
class DataTransformationConfig:
    artifact_dir: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str

    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
        self.transformed_test_file_path = os.path.join(artifact_dir, 'test.npy')
        self.transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, feature_store_file_path, artifact_folder):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig(artifact_folder)
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            # Load CSV data
            data = pd.read_csv(feature_store_file_path)

            # Ensure target column exists
            if TARGET_COLUMN not in data.columns:
                raise CustomException(f"Target column '{TARGET_COLUMN}' not found in the dataset", sys)

            # Rename the target column if needed
            ## data.rename(columns={"Risk": TARGET_COLUMN}, inplace=True)
            return data

        except FileNotFoundError:
            raise CustomException(f"File not found: {feature_store_file_path}", sys)
        except pd.errors.EmptyDataError:
            raise CustomException(f"CSV file is empty or not formatted correctly: {feature_store_file_path}", sys)
        except Exception as e:
            raise CustomException(f"Error reading data from {feature_store_file_path}: {e}", sys)

    def get_data_transformer_object(self, numeric_cols):
        try:
            # Create a pipeline for missing value imputation and scaling
            imputer_step = ('imputer', KNNImputer(n_neighbors=5))
            scaler_step = ('scaler', StandardScaler())

            # Return the combined pipeline
            preprocessor = Pipeline(steps=[imputer_step, scaler_step])
            return preprocessor

        except Exception as e:
            raise CustomException(f"Error creating data transformer object: {e}", sys)

    def initiate_data_transformation(self):
        logging.info("Starting data transformation process.")

        try:
            # Load the dataset from the feature store
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)

            # Check for required columns
            required_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose', 'Age', 'Credit amount', 'Duration']
            for col in required_cols:
                if col not in dataframe.columns:
                    raise CustomException(f"Missing required column: {col}", sys)

            # Fill missing values for categorical columns with the mode (most frequent value)
            dataframe['Saving accounts'].fillna(dataframe['Saving accounts'].mode()[0], inplace=True)
            dataframe['Checking account'].fillna(dataframe['Checking account'].mode()[0], inplace=True)

            # Label encoding for categorical columns
            label_encoding_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
            label_encoder = LabelEncoder()
            for col in label_encoding_cols:
                dataframe[col] = label_encoder.fit_transform(dataframe[col])

            # Separate features and target
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN].map({"good": 1, "bad": 0})  # Map target to 1 and 0

            # Apply SMOTE to address class imbalance
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            # Define numeric columns for scaling
            numeric_cols = ['Age', 'Credit amount', 'Duration']

            # Get the preprocessing pipeline
            preprocessor = self.get_data_transformer_object(numeric_cols=numeric_cols)

            # Apply preprocessing to the training and test sets
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Save the preprocessor pipeline object for later use
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)

            # Combine scaled data with the target values
            train_arr = np.c_[X_train_scaled, y_train]
            test_arr = np.c_[X_test_scaled, y_test]

            logging.info("Data transformation completed successfully.")
            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            raise CustomException(f"Error during data transformation: {e}", sys)
