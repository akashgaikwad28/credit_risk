import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    
    def __init__(self, artifact_folder: str):
        """
        Initializes the TrainingPipeline class with the given artifact folder path.
        """
        self.artifact_folder = artifact_folder

    def start_data_ingestion(self):
        """
        Starts the data ingestion process.
        Returns the path where the feature store file is saved.
        """
        try:
            logging.info("Starting data ingestion process.")
            data_ingestion = DataIngestion(self.artifact_folder)  # Pass artifact folder to DataIngestion
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Data saved at: {feature_store_file_path}")
            return feature_store_file_path
        except Exception as e:
            logging.error(f"Error occurred in data ingestion: {str(e)}")
            raise CustomException(e, sys)

    def start_data_transformation(self, feature_store_file_path):
        """
        Starts the data transformation process, taking the ingested data file path as input.
        """
        try:
            logging.info("Starting data transformation process.")
            # Pass the feature store file path and artifact folder to the DataTransformation class
            data_transformation = DataTransformation(
                feature_store_file_path=feature_store_file_path, 
                artifact_folder=self.artifact_folder
            )
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            logging.error(f"Error occurred in data transformation: {str(e)}")
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):
        """
        Starts the model training process, taking transformed training and testing data.
        """
        try:
            logging.info("Starting model training process.")
            # Initialize ModelTrainer with artifact folder
            model_trainer = ModelTrainer()  # Pass artifact folder to ModelTrainer
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed with score: {model_score}")
            return model_score
        except Exception as e:
            logging.error(f"Error occurred in model training: {str(e)}")
            raise CustomException(e, sys)

    def run_pipeline(self):
        """
        Runs the entire pipeline: Data Ingestion -> Data Transformation -> Model Training.
        """
        try:
            # Step 1: Data Ingestion
            feature_store_file_path = self.start_data_ingestion()

            # Step 2: Data Transformation
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(feature_store_file_path)

            # Step 3: Model Training
            model_score = self.start_model_training(train_arr, test_arr)

            # Output the model score
            logging.info(f"Training pipeline completed successfully. Model score: {model_score}")
            print(f"Training completed. Trained model score: {model_score}")

        except Exception as e:
            logging.error(f"Error occurred during pipeline execution: {str(e)}")
            raise CustomException(e, sys)


# Example usage:
if __name__ == "__main__":
    try:
        # Specify the artifact folder where the intermediate files (like data, models, etc.) will be stored
        artifact_folder = "artifacts"  # Update with the correct folder path
        
        # Initialize the pipeline with the artifact folder
        pipeline = TrainingPipeline(artifact_folder)
        
        # Run the entire training pipeline
        pipeline.run_pipeline()

    except Exception as e:
        logging.error(f"Error occurred during pipeline execution: {str(e)}")
