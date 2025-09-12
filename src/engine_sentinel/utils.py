import os
import sys
import dill
import logging

from src.engine_sentinel.exception import CustomException

def save_object(file_path, obj):
    """
    This function saves a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully to: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    This function loads a Python object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)