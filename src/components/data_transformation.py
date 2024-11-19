from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

TARGET_COLUMN = 'Target'  

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
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, categorical_cols, numeric_cols):
        try:
            imputer_step = ('imputer', KNNImputer(n_neighbors=5))
            scaler_step = ('scaler', StandardScaler())

            preprocessor = Pipeline(
                steps=[imputer_step, scaler_step]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info("Starting data transformation process.")

        try:
            # Load data
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)

            # Impute missing values
            dataframe['Saving accounts'].fillna(dataframe['Saving accounts'].mode()[0], inplace=True)
            dataframe['Checking account'].fillna(dataframe['Checking account'].mode()[0], inplace=True)

            # Label encoding
            label_encoding_cols = ['Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age group']
            label_encoder = LabelEncoder()
            for col in label_encoding_cols:
                dataframe[col] = label_encoder.fit_transform(dataframe[col])

            # Separate features and target
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)

            # Address class imbalance using SMOTE
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            # Data scaling
            numeric_cols = ['Age', 'Credit amount', 'Duration']
            preprocessor = self.get_data_transformer_object(categorical_cols=[], numeric_cols=numeric_cols)

            # Preprocessing pipeline
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Save preprocessor object
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)

            # Combine scaled data and target
            train_arr = np.c_[X_train_scaled, y_train]
            test_arr = np.c_[X_test_scaled, y_test]

            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
