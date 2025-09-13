import sys
import pandas as pd
from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.utils import load_object
from src.engine_sentinel.logger import logger
import os

class PredictionPipeline:
    """
    This class is responsible for loading the trained model and making predictions.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Loads the model and makes a prediction on the input features.
        
        Args:
            features (pd.DataFrame): The input data for which to make predictions.
            
        Returns:
            np.ndarray: The predicted RUL value(s) in a numpy array.
        """
        try:
            # Construct the absolute path to the model file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, 'artifacts', 'model.pkl')

            # Check if the file exists before attempting to load
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            model = load_object(file_path=model_path)
            logger.info("Trained model loaded successfully for prediction.")
            
            preds = model.predict(features)
            logger.info(f"Prediction successful. Predicted RUL: {preds[0]}")
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping input data from a web form
    to the DataFrame format that our model expects.
    """
    def __init__(self, op_setting_1: float, op_setting_2: float, op_setting_3: float, 
                 sensor_1: float, sensor_2: float, sensor_3: float, sensor_4: float, 
                 sensor_5: float, sensor_6: float, sensor_7: float, sensor_8: float, 
                 sensor_9: float, sensor_10: float, sensor_11: float, sensor_12: float, 
                 sensor_13: float, sensor_14: float, sensor_15: float, sensor_16: float, 
                 sensor_17: float, sensor_18: float, sensor_19: float, sensor_20: float, 
                 sensor_21: float):
        
        # --- THIS IS THE COMPLETED SECTION ---
        # Assign all input values from the form to instance variables
        self.op_setting_1 = op_setting_1
        self.op_setting_2 = op_setting_2
        self.op_setting_3 = op_setting_3
        self.sensor_1 = sensor_1
        self.sensor_2 = sensor_2
        self.sensor_3 = sensor_3
        self.sensor_4 = sensor_4
        self.sensor_5 = sensor_5
        self.sensor_6 = sensor_6
        self.sensor_7 = sensor_7
        self.sensor_8 = sensor_8
        self.sensor_9 = sensor_9
        self.sensor_10 = sensor_10
        self.sensor_11 = sensor_11
        self.sensor_12 = sensor_12
        self.sensor_13 = sensor_13
        self.sensor_14 = sensor_14
        self.sensor_15 = sensor_15
        self.sensor_16 = sensor_16
        self.sensor_17 = sensor_17
        self.sensor_18 = sensor_18
        self.sensor_19 = sensor_19
        self.sensor_20 = sensor_20
        self.sensor_21 = sensor_21
        # --- END OF COMPLETED SECTION ---

    def get_data_as_dataframe(self):
        """
        Converts the custom input data into a single-row pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                "op_setting_1": [self.op_setting_1],
                "op_setting_2": [self.op_setting_2],
                "op_setting_3": [self.op_setting_3],
                "sensor_1": [self.sensor_1],
                "sensor_2": [self.sensor_2],
                "sensor_3": [self.sensor_3],
                "sensor_4": [self.sensor_4],
                "sensor_5": [self.sensor_5],
                "sensor_6": [self.sensor_6],
                "sensor_7": [self.sensor_7],
                "sensor_8": [self.sensor_8],
                "sensor_9": [self.sensor_9],
                "sensor_10": [self.sensor_10],
                "sensor_11": [self.sensor_11],
                "sensor_12": [self.sensor_12],
                "sensor_13": [self.sensor_13],
                "sensor_14": [self.sensor_14],
                "sensor_15": [self.sensor_15],
                "sensor_16": [self.sensor_16],
                "sensor_17": [self.sensor_17],
                "sensor_18": [self.sensor_18],
                "sensor_19": [self.sensor_19],
                "sensor_20": [self.sensor_20],
                "sensor_21": [self.sensor_21],

            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logger.info("Custom input data converted to DataFrame successfully.")
            return df

        except Exception as e:
            raise CustomException(e, sys)
