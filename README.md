## END TO END MACHINE LEARNING PROJECT
EngineSentinel: Predictive Maintenance for Turbofan Engines ✈️
A complete, end-to-end Machine Learning project to predict the Remaining Useful Life (RUL) of turbofan engines, built with a production-ready mindset.

📋 Table of Contents
Project Overview

Tech Stack

Project Architecture

Workflow

How to Run

Results

Future Work

🚀 Project Overview
EngineSentinel is a predictive maintenance solution that leverages machine learning to forecast the failure of aircraft turbofan engines. By analyzing time-series sensor data provided by NASA, this project builds an automated pipeline to train a model that predicts the Remaining Useful Life (RUL) in operational cycles.

The primary goal is to shift from a reactive or preventative maintenance schedule to a predictive one, which can lead to significant cost savings, increased operational efficiency, and enhanced safety.

💻 Tech Stack
This project utilizes a modern, scalable tech stack, common in production MLOps environments:

Category

Technologies

Programming

Python 3.13

Data Handling

Pandas, NumPy

ML Modeling

Scikit-learn, XGBoost, LightGBM

Experiment Tracking

MLflow, DagsHub

Project Structure

Virtual Environments (venv), Modular Packaging (setup.py)

CI/CD & Automation

GitHub Actions (planned)

Deployment

FastAPI (for prediction API), Docker (planned), AWS/Azure (planned)

🏗️ Project Architecture
The project is built around a modular, end-to-end pipeline that automates the entire machine learning workflow, from raw data to a trained and evaluated model.

A high-level overview of the training pipeline flow.

🔄 Workflow
The project is broken down into a series of modular components, orchestrated by a central pipeline script.

Data Ingestion:

Reads the raw .txt files from the NASA dataset.

Assigns correct column headers based on the documentation.

Saves the cleaned, raw data as a .csv artifact for downstream use.

Data Transformation (Feature Engineering):

Loads the raw data artifact.

Calculates the crucial Remaining Useful Life (RUL) for each data point by analyzing the time-to-failure for each engine.

Saves the final, enriched dataset (with the RUL target variable) as a new artifact.

Model Training:

Loads the transformed, feature-engineered data.

Splits the data into training and testing sets.

Trains the best-performing model identified during experimentation (LightGBM).

Saves the trained model as a model.pkl pickle file.

Model Evaluation:

Loads the saved model and the test dataset.

Calculates the R² score to measure the model's performance on unseen data.

Logs the performance metric to an experiment tracking server (MLflow) for versioning and comparison.

▶️ How to Run
Follow these steps to set up the environment and run the training pipeline.

Prerequisites:

Python 3.13

Git

Homebrew (for macOS users, to install libomp)

1. Clone the Repository:

git clone [https://github.com/DHRUVBSTHAKUR/EngineSentinel.git](https://github.com/DHRUVBSTHAKUR/EngineSentinel.git)
cd EngineSentinel

2. Create and Activate a Virtual Environment:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies:
For macOS users, first install libomp:

brew install libomp

Then, install all required Python packages:

pip install -r requirements.txt

4. Install the Project as a Local Package:
This step makes all your source code importable.

pip install -e .

5. Run the Training Pipeline:
Execute the main pipeline script. This will run all steps from data ingestion to model evaluation.

python -m engine_sentinel.pipeline.train_pipeline

After running, you will find all generated files (logs, datasets, and the trained model.pkl) in the logs/ and artifacts/ directories.

📈 Results
The experimentation phase compared three models: Random Forest, XGBoost, and LightGBM. The final automated pipeline trains the LightGBM Regressor, which achieved a strong baseline performance:

R² Score on Test Set: ~0.62

This indicates that the model can explain approximately 62% of the variance in the engine's RUL, providing a solid foundation for a predictive maintenance system.

🔮 Future Work
This project provides a robust foundation. Future improvements could include:

Hyperparameter Tuning: Use tools like Optuna or GridSearchCV to find the optimal settings for the LightGBM model to improve its accuracy.

Web Application: Build a user-friendly web interface using FastAPI or Flask that uses the prediction_pipeline.py to allow users to input sensor data and get a real-time RUL prediction.

Containerization: Dockerize the entire application for easy, reproducible deployment.

Cloud Deployment: Deploy the prediction API as a serverless function on AWS Lambda or Azure Functions for a scalable, cost-effective solution.