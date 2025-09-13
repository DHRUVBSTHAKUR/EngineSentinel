import sys
from pathlib import Path
import os

# --- THE FIX IS HERE ---
# This block adds your project's 'src' directory to Python's path
# It makes your 'engine_sentinel' package importable
project_root = Path(__file__).resolve().parent
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
print(f"DEBUG: Added the following path to sys.path -> {src_path}") # For verification
# --- END OF FIX ---

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# Corrected the import statement to match the file name 'predict_pipeline.py'
from engine_sentinel.pipeline.predict_pipeline import CustomData, PredictionPipeline

# Create a Flask application instance
application = Flask(__name__)

# For convenience, we can use 'app' as well
app = application

@app.route('/')
def index():
    """
    Renders the home page with the input form.
    """
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles the prediction logic.
    """
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            # Create a CustomData object from the form input
            data = CustomData(
                op_setting_1=float(request.form.get('op_setting_1')),
                op_setting_2=float(request.form.get('op_setting_2')),
                op_setting_3=float(request.form.get('op_setting_3')),
                sensor_1=float(request.form.get('sensor_1')),
                sensor_2=float(request.form.get('sensor_2')),
                sensor_3=float(request.form.get('sensor_3')),
                sensor_4=float(request.form.get('sensor_4')),
                sensor_5=float(request.form.get('sensor_5')),
                sensor_6=float(request.form.get('sensor_6')),
                sensor_7=float(request.form.get('sensor_7')),
                sensor_8=float(request.form.get('sensor_8')),
                sensor_9=float(request.form.get('sensor_9')),
                sensor_10=float(request.form.get('sensor_10')),
                sensor_11=float(request.form.get('sensor_11')),
                sensor_12=float(request.form.get('sensor_12')),
                sensor_13=float(request.form.get('sensor_13')),
                sensor_14=float(request.form.get('sensor_14')),
                sensor_15=float(request.form.get('sensor_15')),
                sensor_16=float(request.form.get('sensor_16')),
                sensor_17=float(request.form.get('sensor_17')),
                sensor_18=float(request.form.get('sensor_18')),
                sensor_19=float(request.form.get('sensor_19')),
                sensor_20=float(request.form.get('sensor_20')),
                sensor_21=float(request.form.get('sensor_21'))
            )
            
            # Convert the data to a DataFrame for prediction
            pred_df = data.get_data_as_dataframe()
            
            # Use the prediction pipeline to get the RUL
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Render the results page with the prediction
            return render_template('results.html', results=round(results[0], 2))
        except Exception as e:
            # This will print the specific error to your terminal
            print(f"An error occurred during prediction: {e}")
            return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
