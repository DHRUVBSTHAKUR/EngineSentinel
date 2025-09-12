import os
import sys
from dataclasses import dataclass
import pandas as pd

from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.logger import logger

@dataclass
class DataIngestionConfig:
    """
    Configuration class for the data ingestion component.
    Defines paths for raw, train, and test data artifacts.
    """
    raw_data_path: str = os.path.join('artifacts', "raw_data.csv")
    # We aren't creating a train/test split here, but this is good practice
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataIngestion:
    """
    This class handles reading the raw data from the source,
    assigning column names, and saving it as a clean CSV file.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Main method to perform data ingestion.
        """
        logger.info("Entered the data ingestion method or component")
        try:
            # Define column names based on the dataset documentation
            column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                           [f'sensor_{i}' for i in range(1, 22)]

            # --- This logic is directly from your notebook ---
            # Load the training data from the data folder
            df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None)
            
            # Drop the extra empty columns
            df.drop(columns=[26, 27], inplace=True)

            # Assign the correct column names
            df.columns = column_names
            # --- End of notebook logic ---
            
            logger.info("Read the dataset as dataframe")

            # Create the 'artifacts' directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data with correct headers
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")
            
            # For this project, we can consider the raw data as our "training" set for now
            # In a more complex project, we would also create a test split here.
            
            logger.info("Ingestion of the data is completed")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


