from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os
import sys
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

# Initialize Flask app
app = Flask(__name__)

# Define the artifact folder path (relative to the project directory)
artifact_folder = os.path.join(os.getcwd(), "artifacts")

@app.route("/")
def home():
    return "Welcome to my application"

# Training route
@app.route("/train")
def train_route():
    try:
        # Pass the artifact folder path to TrainingPipeline
        train_pipeline = TrainingPipeline(artifact_folder=artifact_folder)
        train_pipeline.run_pipeline()
        return "Training Completed."
    except Exception as e:
        raise CustomException(e, sys)

# Prediction route (handles both GET and POST)
@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            # Check if the file is part of the request
            if 'file' not in request.files:
                return "No file part in the request.", 400
            
            uploaded_file = request.files['file']

            # Ensure the file is present and has a valid name
            if uploaded_file.filename == '':
                return "No selected file.", 400

            # Check if the file has a valid CSV extension and MIME type
            if uploaded_file.filename.endswith('.csv') or uploaded_file.content_type == 'text/csv':
                # Process the file using PredictionPipeline
                prediction_pipeline = PredictionPipeline(request)
                prediction_file_detail = prediction_pipeline.run_pipeline()

                lg.info(f"Prediction completed for file: {uploaded_file.filename}.")
                return send_file(prediction_file_detail.prediction_file_path,
                                 download_name=prediction_file_detail.prediction_file_name,
                                 as_attachment=True)
            else:
                return "Please upload a valid CSV file.", 400
        else:
            # Render the file upload HTML page
            return render_template('upload_file.html')
    
    except Exception as e:
        # Log the error
        lg.error(f"Error during prediction: {e}")
        raise CustomException(e, sys)


# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
