import os
import sys
from dataclasses import dataclass

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.logger import logger
from src.engine_sentinel.utils import save_object

@dataclass
class ModelTrainerConfig:
    """
    Configuration for the model training component.
    Specifies the save path for the trained model artifact.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    Handles the model training, evaluation, and saving process.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, transformed_data_path: str):
        """
        Orchestrates the model training process.
        
        Args:
            transformed_data_path (str): Path to the processed data from the transformation step.
        """
        try:
            logger.info("Starting model training process")
            df = pd.read_csv(transformed_data_path)
            logger.info(f"Loaded transformed data from {transformed_data_path}")

            logger.info("Separating features (X) and target (y)")
            y = df['RUL']
            X = df.drop(columns=['RUL', 'unit_number', 'time_in_cycles'])

            logger.info("Splitting data into training and testing sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # We are using LightGBM as it was one of the best performers in our notebook
            lgbm = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            logger.info("Training LightGBM model...")
            lgbm.fit(X_train, y_train)
            logger.info("Model training complete")
            
            # Evaluate the model on the test set
            y_pred = lgbm.predict(X_test)
            score = r2_score(y_test, y_pred)
            logger.info(f"Model evaluated. R² Score on test set: {score:.2f}")

            logger.info(f"Saving the trained model to: {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=lgbm
            )
            
            logger.info("Trained model object saved successfully")
            
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)

