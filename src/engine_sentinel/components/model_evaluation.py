import os
import sys
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mlflow.models.signature import infer_signature  # <-- NEW IMPORT

from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.logger import logger
from src.engine_sentinel.utils import load_object

class ModelEvaluation:
    """
    This class handles the evaluation of the trained model against the test set
    and logs the performance metrics using MLflow.
    """
    def initiate_model_evaluation(self, transformed_data_path: str, trained_model_path: str):
        """
        Loads the test data and trained model, calculates metrics, and logs them.
        
        Args:
            transformed_data_path (str): Path to the transformed data file.
            trained_model_path (str): Path to the saved, trained model file (.pkl).
        """
        try:
            logger.info("--- Model Evaluation Component Started ---")
            
            df = pd.read_csv(transformed_data_path)
            y = df['RUL']
            X = df.drop(columns=['RUL', 'unit_number', 'time_in_cycles'])
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = load_object(file_path=trained_model_path)
            logger.info("Trained model loaded successfully.")

            mlflow.set_registry_uri("https://dagshub.com/DHRUVBSTHAKUR/EngineSentinel.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                logger.info("Making predictions on the test set.")
                predicted_rul = model.predict(X_test)
                
                r2 = r2_score(y_test, predicted_rul)
                logger.info(f"Final R² score on test data: {r2:.2f}")

                mlflow.log_metric("r2_score", r2)

                # --- FIX IS HERE: Add signature and input example ---
                input_example = X_test.head(1)
                signature = infer_signature(input_example, predicted_rul[:1])
                # --- END OF FIX ---

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        sk_model=model, 
                        artifact_path="model", 
                        signature=signature, # <-- NEW
                        input_example=input_example, # <-- NEW
                        registered_model_name="LGBMRegressor"
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=model, 
                        artifact_path="model",
                        signature=signature, # <-- NEW
                        input_example=input_example # <-- NEW
                    )

            logger.info("--- Model Evaluation Component Completed ---")

        except Exception as e:
            raise CustomException(e, sys)

