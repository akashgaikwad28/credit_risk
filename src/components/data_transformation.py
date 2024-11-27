import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
import pickle
import logging

import category_encoders as ce  # For Binary Encoding
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging
log_file = "data_transformation.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
)
logger = logging.getLogger()

@dataclass
class DataTransformationConfig:
    def __init__(self, artifact_folder="artifacts"):
        self.artifact_dir = artifact_folder
        self.transformed_train_file_path = os.path.join(self.artifact_dir, 'train.npy')
        self.transformed_test_file_path = os.path.join(self.artifact_dir, 'test.npy')
        self.transformed_object_file_path = os.path.join(self.artifact_dir, 'preprocessor.pkl')


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for label encoding multiple columns"""
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {col: LabelEncoder() for col in columns}

    def fit(self, X, y=None):
        for col in self.columns:
            self.label_encoders[col].fit(X[col])
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.label_encoders[col].transform(X_copy[col])
        return X_copy


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if col in self.columns:
                X_copy[col] = X_copy[col].apply(lambda x: 1 if x == 'good' or x == 'male' else 0)
        return X_copy



class DataTransformation:
    def __init__(self, feature_store_file_path, artifact_folder=None):
        """
        Initializes the DataTransformation class.
        :param feature_store_file_path: Path to the ingested feature store.
        :param artifact_folder: Path to the folder where artifacts are stored (optional).
        """
        self.feature_store_file_path = feature_store_file_path
        self.artifact_folder = artifact_folder if artifact_folder else "artifacts"  # Default to 'artifacts'
        self.data_transformation_config = DataTransformationConfig(self.artifact_folder)

    def get_data(self):
        """Reads and preprocesses the data."""
        try:
            logger.info("Loading dataset from: %s", self.feature_store_file_path)
            df = pd.read_csv(self.feature_store_file_path)

            logger.info("Dropping unnecessary columns (if any).")
            if "Unnamed: 0" in df.columns:
                df.drop(["Unnamed: 0"], axis=1, inplace=True)

            logger.info("Handling missing values.")
            df.fillna(method='ffill', inplace=True)  # Forward fill as an example

            label_encoding_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
            label_encoder = LabelEncoderTransformer(columns=label_encoding_cols)
            df = label_encoder.fit_transform(df)

            logger.info("Returning preprocessed dataframe.")
            return df
        except Exception as e:
            logger.error("Error during data loading: %s", e)
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """Creates and returns a data transformer pipeline with imputation, scaling, and encoding."""
        try:
            logger.info("Creating data transformation pipeline.")
            

            # Column names for label encoding
            label_encode_columns = ['Housing', 'Saving accounts', 'Checking account', 'Purpose']

            # Define the pipeline steps
            

            pipeline_steps = [
                ('label_encoder', LabelEncoderTransformer(columns=label_encode_columns)),
                ('binary_encoder', BinaryEncoder(columns=['Sex', 'Risk'])),
                ('imputer', SimpleImputer(strategy='mean')),  # Use SimpleImputer
                ('scaler', RobustScaler())
            ]



            pipeline = Pipeline(pipeline_steps)
            return pipeline
        except Exception as e:
            logger.error("Error during pipeline creation: %s", e)
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """Main function for data transformation."""
        try:
            logger.info("Starting data transformation process.")
            dataframe = self.get_data()

            logger.info("Splitting features and target variable.")
            X = dataframe.drop(columns=["Risk"])
            y = np.where(dataframe["Risk"] == "good", 1, 0)

            logger.info("Splitting the dataset into train and test sets.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            logger.info("Applying preprocessing pipeline to training data.")
            preprocessor = self.get_data_transformer_object()
            X_train_processed = preprocessor.fit_transform(X_train)
            
            # # Apply the pipeline to the training data
            # X_train_processed = preprocessor.fit_transform(X_train)

            # # Convert the transformed array back to a DataFrame with original column names
            # X_train_processed_df = pd.DataFrame(X_train_processed, columns=X_train.columns)


            logger.info("Applying SMOTE to handle class imbalance.")
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

            logger.info("Applying preprocessing pipeline to testing data.")
            X_test_processed = preprocessor.transform(X_test)

            logger.info("Combining processed data with target variable.")
            train_arr = np.c_[X_train_smote, y_train_smote]
            test_arr = np.c_[X_test_processed, y_test]

            logger.info("Saving preprocessor object to file.")
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logger.info("Data transformation completed successfully.")
            return (train_arr, test_arr, preprocessor_path)
        except Exception as e:
            logger.error("Error during data transformation: %s", e)
            raise CustomException(e, sys)
