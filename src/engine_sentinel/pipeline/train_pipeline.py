import sys
from src.engine_sentinel.exception import CustomException
from src.engine_sentinel.logger import logger
from src.engine_sentinel.components.data_ingestion import DataIngestion
from src.engine_sentinel.components.data_transformation import DataTransformation
from src.engine_sentinel.components.model_trainer import ModelTrainer
from src.engine_sentinel.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    """
    This class orchestrates the entire training process by calling
    each component in the correct order.
    """
    def run_pipeline(self):
        """
        Runs the end-to-end training pipeline.
        """
        try:
            logger.info("--- Training Pipeline Started ---")
            
            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            raw_data_path = data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            transformed_data_path = data_transformation.initiate_data_transformation(raw_data_path=raw_data_path)
            
            # Step 3: Model Training
            model_trainer = ModelTrainer()
            trained_model_path = model_trainer.initiate_model_trainer(transformed_data_path=transformed_data_path)

            # Step 4: Model Evaluation (NEW STEP)
            model_evaluation = ModelEvaluation()
            model_evaluation.initiate_model_evaluation(
                transformed_data_path=transformed_data_path,
                trained_model_path=trained_model_path
            )
            
            logger.info(f"--- Training Pipeline Completed Successfully ---")
            logger.info(f"Trained model saved at: {trained_model_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

