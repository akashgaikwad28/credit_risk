import os
import pandas as pd
import logging
import sys
from dotenv import load_dotenv
from flask import request
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass
from datetime import datetime
from flask import request 
import logging

# Load environment variables
load_dotenv()

TARGET_COLUMN = os.getenv("TARGET_COLUMN")
if not TARGET_COLUMN:
    raise CustomException("TARGET_COLUMN environment variable is not set.", sys)

# Dataclass for the Prediction Pipeline Configuration
@dataclass
class PredictionPipelineConfig:
    artifact_folder: str = os.getenv("ARTIFACT_FOLDER")
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "prediction_file.csv"
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)

# PredictionPipeline class
class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()

        # Load preprocessor and model
        self.model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
        self.preprocessor = self.utils.load_object(self.prediction_pipeline_config.preprocessor_path)

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting data  process data by preprocessor in predict file.")
            # Apply the same preprocessing pipeline to the input features
            transformed_x = self.preprocessor.fit_transform(features)

            # Make predictions
            preds = self.model.predict(transformed_x)
            return preds
        except Exception as e:
            raise CustomException(f"Error during prediction: {str(e)}", sys)

    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            prediction_column_name: str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)

            # Drop unwanted columns
            if "Unnamed: 0" in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(columns="Unnamed: 0")

            predictions = self.predict(input_dataframe)

            # Add predictions to the dataframe
            input_dataframe[prediction_column_name] = [pred for pred in predictions]

            # Mapping target column to 'bad' or 'good'
            target_column_mapping = {0: 'bad', 1: 'good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            # Save predicted dataframe
            unique_filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.prediction_pipeline_config.prediction_file_path = os.path.join(self.prediction_pipeline_config.prediction_output_dirname, unique_filename)
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)

            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)

        except Exception as e:
            raise CustomException(f"Error in generating predicted dataframe: {str(e)}", sys)

    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()  # Save the input CSV file
            self.get_predicted_dataframe(input_csv_path)  # Generate predictions

            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(f"Error in running prediction pipeline: {str(e)}", sys)

    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            if 'file' not in self.request.files:  # Corrected to self.request
                raise CustomException("No file part in the request", sys)
            
            input_csv_file = self.request.files['file']  # Corrected to self.request
            
            if input_csv_file.filename == '':
                raise CustomException("No selected file", sys)

            if not input_csv_file.filename.endswith('.csv'):
                raise CustomException("Only CSV files are allowed.", sys)

            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)

            input_csv_file.save(pred_file_path)
            logging.info(f"File saved at {pred_file_path}")

            return pred_file_path
        except Exception as e:
            raise CustomException(f"Error saving input file: {str(e)}", sys)
    # def predict(self, features: pd.DataFrame):
    #     try:
    #         model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
    #         preprocessor = self.utils.load_object(file_path=self.prediction_pipeline_config.preprocessor_path)

    #         logging.info(f"Transforming input features using preprocessor.")
    #         transformed_x = preprocessor.transform(features)

    #         logging.info(f"Making predictions using model.")
    #         preds = model.predict(transformed_x)

    #         return preds
    #     except Exception as e:
    #         raise CustomException(f"Error during prediction: {str(e)}", sys)
   
    # def get_predicted_dataframe(self, input_dataframe_path: str):
    #     try:
    #         prediction_column_name: str = TARGET_COLUMN
    #         input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)

    #         logging.info(f"Reading input dataframe from {input_dataframe_path}")

    #         # Check and drop unwanted columns (e.g., Unnamed: 0 column)
    #         if "Unnamed: 0" in input_dataframe.columns:
    #             input_dataframe = input_dataframe.drop(columns="Unnamed: 0")

    #         # # Check for missing columns
    #         # expected_columns = ['Saving accounts', 'Risk', 'Checking account', 'Housing', 'Purpose', 'Age', 'Credit amount', 'Duration']  # Example of expected columns
    #         # missing_columns = [col for col in expected_columns if col not in input_dataframe.columns]
    #         # if missing_columns:
    #         #     raise CustomException(f"Missing expected columns: {', '.join(missing_columns)}", sys)

    #         predictions = self.predict(input_dataframe)

    #         # Add predictions to the dataframe
    #         input_dataframe[prediction_column_name] = [pred for pred in predictions]

    #         # Mapping target column to 'bad' or 'good'
    #         target_column_mapping = {0: 'bad', 1: 'good'}
    #         input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

    #         # Generate unique filename for predictions
    #         unique_filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    #         self.prediction_pipeline_config.prediction_file_path = os.path.join(self.prediction_pipeline_config.prediction_output_dirname, unique_filename)

    #         # Save the result to file
    #         os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)
    #         input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)

    #         logging.info(f"Predictions saved to {self.prediction_pipeline_config.prediction_file_path}")

    #     except Exception as e:
    #         raise CustomException(f"Error in generating predicted dataframe: {str(e)}", sys)

    # def run_pipeline(self):
    #     try:
    #         logging.info("Starting prediction pipeline.")
    #         input_csv_path = self.save_input_files()
    #         self.get_predicted_dataframe(input_csv_path)

    #         logging.info("Prediction pipeline completed successfully.")
    #         return self.prediction_pipeline_config
    #     except Exception as e:
    #         raise CustomException(f"Error in running prediction pipeline: {str(e)}", sys)
