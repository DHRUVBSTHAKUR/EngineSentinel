import os
import sys
from dataclasses import dataclass

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.logger import logger
from src.engine_sentinel.utils import save_object, LGBM_PARAM_GRID # <-- UPDATED IMPORT

@dataclass
class ModelTrainerConfig:
    """
    Configuration for the model training component.
    Specifies the save path for the trained model artifact.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    Handles model training with hyperparameter tuning, evaluation, and saving.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, transformed_data_path: str):
        """
        Orchestrates the model training and tuning process.
        
        Args:
            transformed_data_path (str): Path to the processed data from the transformation step.
        """
        try:
            logger.info("--- Model Trainer Component Started ---")
            df = pd.read_csv(transformed_data_path)
            logger.info(f"Loaded transformed data from {transformed_data_path}")

            logger.info("Separating features (X) and target (y)")
            y = df['RUL']
            X = df.drop(columns=['RUL', 'unit_number', 'time_in_cycles'])

            logger.info("Splitting data into training and testing sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # --- HYPERPARAMETER TUNING SETUP ---
            lgbm = LGBMRegressor(random_state=42, n_jobs=-1)

            # The parameter grid is now imported from utils.py
            param_grid = LGBM_PARAM_GRID

            # Create the GridSearchCV object
            # cv=3 means 3-fold cross-validation
            grid_search = GridSearchCV(
                estimator=lgbm,
                param_grid=param_grid,
                cv=3,
                scoring='r2',
                verbose=1,
                n_jobs=-1
            )
            
            logger.info("Starting Hyperparameter Tuning (Grid Search)...")
            # This will train the model multiple times to find the best settings
            grid_search.fit(X_train, y_train)
            logger.info("Hyperparameter Tuning complete.")

            # Get the best model found by the grid search
            best_lgbm = grid_search.best_estimator_
            
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            
            # --- END OF HYPERPARAMETER TUNING ---

            # Evaluate the BEST model on the test set
            y_pred = best_lgbm.predict(X_test)
            score = r2_score(y_test, y_pred)
            logger.info(f"Tuned model evaluated. R² Score on test set: {score:.2f}")

            logger.info(f"Saving the best trained model to: {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_lgbm  # Save the best model
            )
            
            logger.info("Trained model object saved successfully")
            
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)

