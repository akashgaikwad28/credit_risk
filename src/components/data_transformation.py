import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
import pickle
import logging

# Set up logging
log_file = "data_transformation.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print logs to console
        logging.FileHandler(log_file)  # Write logs to file
    ]
)
logger = logging.getLogger()

@dataclass
class DataTransformationConfig:
    def __init__(self, artifact_folder="artifacts"):
        self.artifact_dir = artifact_folder
        self.transformed_train_file_path = os.path.join(self.artifact_dir, 'train.npy')
        self.transformed_test_file_path = os.path.join(self.artifact_dir, 'test.npy')
        self.transformed_object_file_path = os.path.join(self.artifact_dir, 'preprocessor.pkl')


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
            # Load dataset
            df = pd.read_csv(self.feature_store_file_path)

            logger.info("Dropping unnecessary columns (if any).")
            # Drop unnecessary columns
            if "Unnamed: 0" in df.columns:
                df.drop(["Unnamed: 0"], axis=1, inplace=True)

            # Handle missing values if needed (before encoding)
            logger.info("Handling missing values.")
            df.fillna(method='ffill', inplace=True)  # Forward fill as an example

            # Encode specific categorical variables
            logger.info("Encoding categorical variables.")
            saving_accounts_mapping = {'little': 0, 'moderate': 1, 'rich': 2, 'quite rich': 3}
            checking_account_mapping = {'little': 0, 'moderate': 1, 'rich': 2}
            df['Saving accounts'] = df['Saving accounts'].map(saving_accounts_mapping)
            df['Checking account'] = df['Checking account'].map(checking_account_mapping)

            # Columns to apply Label Encoding
            label_encoding_cols = ['Sex','Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
            # Initialize Label Encoder
            label_encoder = LabelEncoder()

            # Apply Label Encoding to each column
            for col in label_encoding_cols:
                if df[col].isnull().sum() > 0:
                    logger.warning("Column %s contains missing values. These will be handled before encoding.", col)
                    df[col].fillna(df[col].mode()[0], inplace=True)  # Fill missing values with mode

                logger.info(f"Encoding column {col}.")
                df[col] = label_encoder.fit_transform(df[col])

            logger.info("Returning preprocessed dataframe.")
            return df
        except Exception as e:
            logger.error("Error during data loading: %s", e)
            raise

    def get_data_transformer_object(self):
        """Creates and returns a data transformer pipeline."""
        try:
            logger.info("Creating data transformation pipeline.")
            # Define a pipeline with imputation and scaling
            pipeline = Pipeline([ 
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', RobustScaler())
            ])
            return pipeline
        except Exception as e:
            logger.error("Error during pipeline creation: %s", e)
            raise

    def initiate_data_transformation(self):
        """Main function for data transformation."""
        try:
            logger.info("Starting data transformation process.")
            
            # Load and preprocess data
            dataframe = self.get_data()

            # Splitting features and target
            logger.info("Splitting features and target variable.")
            X = dataframe.drop(columns=["Risk"])
            y = np.where(dataframe["Risk"] == "good", 1, 0)

            # Splitting the dataset into train-test
            logger.info("Splitting the dataset into train and test sets.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Apply preprocessing
            logger.info("Applying preprocessing steps to training and testing data.")
            preprocessor = self.get_data_transformer_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Apply SMOTE to handle class imbalance
            logger.info("Applying SMOTE for handling class imbalance.")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            logger.info("SMOTE applied. Resampled class distribution in training set: %s", np.bincount(y_train_resampled))

            # Combine processed data with target variable
            logger.info("Combining processed data with target variable.")
            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_scaled, y_test]

            # Save preprocessor
            logger.info("Saving preprocessor object to file.")
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logger.info("Data transformation completed successfully.")
            # Return processed arrays and preprocessor path
            return (train_arr, test_arr, preprocessor_path)
        except Exception as e:
            logger.error("Error during data transformation: %s", e)
            raise
