import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        # Models dictionary based on your YAML configuration
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'StackingClassifier': StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=42)),
                    ('gb', GradientBoostingClassifier(random_state=42)),
                    ('xgb', XGBClassifier(random_state=42))
                ],
                final_estimator=LogisticRegression()
            )
        }

    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            report = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)  # Train model
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
        try:
            model_report: dict = self.evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=self.models)

            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object: object, best_model_name: str, X_train, y_train) -> object:
        try:
            # Reading model configuration from the YAML file
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            print("Best parameters are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting training and testing input and target features")

           
            target_column = 'Risk'

            # Check if train_array and test_array are numpy arrays or pandas DataFrames
            if isinstance(train_array, np.ndarray):
            # If the train array is numpy, convert it to a DataFrame with the feature names
                logging.info("Converting train_array to pandas DataFrame")
                train_array = pd.DataFrame(train_array, columns=['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose','Risk'])

            if isinstance(test_array, np.ndarray):
                # If the test array is numpy, convert it to a DataFrame with the feature names
                logging.info("Converting test_array to pandas DataFrame")
                test_array = pd.DataFrame(test_array, columns=['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose','Risk'])

            # Log the column names to help with debugging
                logging.info(f"Train Array Columns: {train_array.columns}")
                logging.info(f"Test Array Columns: {test_array.columns}")

            # For train array, check if the target column exists
            if target_column in train_array.columns:
                x_train = train_array.drop(columns=[target_column])  # Features (input)
                y_train = train_array[target_column]  # Target variable (Risk)
            else:
                raise Exception(f"Target column '{target_column}' not found in the training data.")

            # For test array, similarly check if 'Risk' exists in the columns
            if target_column in test_array.columns:
                x_test = test_array.drop(columns=[target_column])  # Features (input)
                y_test = test_array[target_column]  # Target variable (Risk)
            else:
                raise Exception(f"Target column '{target_column}' not found in the test data.")

            logging.info(f"Data split completed. Proceeding with model evaluation and training.")

            # Now continue with the model training process as usual
            model_report: dict = self.evaluate_models(X=x_train, y=y_train, models=self.models)

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Get the best model object
            best_model = self.models[best_model_name]

            # Fine-tune the best model
            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )

            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print(f"Best model name: {best_model_name} and score: {best_model_score}")

            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.5")

            logging.info(f"Best found model on both training and testing dataset")

            # Saving the model
            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
