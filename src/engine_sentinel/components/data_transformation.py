import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.logger import logger

@dataclass
class DataTransformationConfig:
    """
    Configuration class for the data transformation component.
    Defines the file path for the output transformed data.
    """
    transformed_data_path: str = os.path.join('artifacts', "transformed_data.csv")

class DataTransformation:
    """
    This class handles the feature engineering part of our pipeline,
    specifically calculating the Remaining Useful Life (RUL).
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, raw_data_path):
        """
        Main method to perform data transformation.
        
        Args:
            raw_data_path (str): The path to the raw data from the data ingestion step.
        """
        try:
            # Load the raw data
            df = pd.read_csv(raw_data_path)
            logger.info("Raw data file loaded successfully for transformation.")

            # --- RUL Calculation Logic (from our notebook) ---
            logger.info("Starting RUL calculation.")
            
            # 1. Find the maximum (last) cycle for each engine
            max_cycles_df = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
            max_cycles_df.columns = ['unit_number', 'max_cycles']
            
            # 2. Merge this 'max_cycles' information back into the original dataframe
            df = pd.merge(df, max_cycles_df, on='unit_number', how='left')
            
            # 3. Calculate the RUL for each row
            df['RUL'] = df['max_cycles'] - df['time_in_cycles']
            
            # 4. Drop the temporary 'max_cycles' column
            df.drop(columns=['max_cycles'], inplace=True)
            
            logger.info("RUL calculation complete.")

            # Save the new, transformed dataframe
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_data_path), exist_ok=True)
            df.to_csv(self.data_transformation_config.transformed_data_path, index=False, header=True)
            
            logger.info(f"Transformed data saved to: {self.data_transformation_config.transformed_data_path}")

            return self.data_transformation_config.transformed_data_path

        except Exception as e:
            raise CustomException(e, sys)

