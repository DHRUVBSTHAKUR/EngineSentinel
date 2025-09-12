import sys
import pandas as pd
from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.utils import load_object
from src.engine_sentinel.logger import logger

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
            float: The predicted RUL value.
        """
        try:
            model_path = 'artifacts/model.pkl'
            model = load_object(file_path=model_path)
            logger.info("Preprocessor and Model loaded successfully for prediction.")
            
            preds = model.predict(features)
            logger.info(f"Prediction successful. Predicted RUL: {preds[0]}")
            return preds[0]

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping input data from a web form or other source
    to the format our model expects.
    """
    def __init__(self, op_setting_1, op_setting_2, op_setting_3, sensor_1, sensor_2,
                 sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8,
                 sensor_9, sensor_10, sensor_11, sensor_12, sensor_13, sensor_14,
                 sensor_15, sensor_16, sensor_17, sensor_18, sensor_19, sensor_20,
                 sensor_21):
        
        # Assign all input values to instance variables
        self.op_setting_1 = op_setting_1
        self.op_setting_2 = op_setting_2
        # ... (assign all other sensor values) ...
        self.sensor_21 = sensor_21


    def get_data_as_dataframe(self):
        """
        Converts the custom input data into a pandas DataFrame.
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
            
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

