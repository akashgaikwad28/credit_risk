from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
import os

# Initialize Flask app
app = Flask(__name__)

# Define the artifact folder path (relative to the project directory)
artifact_folder = os.path.join(os.getcwd(), "artifacts")

@app.route("/")
def home():
    # Render the home page with background image and buttons
    return render_template("home.html")

@app.route("/train", methods=["POST"])
def train_route():
    try:
        
        
        # Trigger the training pipeline
        training_pipeline = TrainingPipeline(artifact_folder=artifact_folder)
        training_pipeline.run_pipeline()
        lg.info("Model training completed successfully.")
        return render_template("train_complete.html")
    except CustomException as e:
        lg.error(f"Error during training: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    if request.method == "POST":
        # Handle file upload and trigger prediction pipeline
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        try:
            # Save uploaded file
            file_path = os.path.join(artifact_folder, file.filename)
            file.save(file_path)

            # Run prediction
            prediction_pipeline = PredictionPipeline()
            prediction_result = prediction_pipeline.run_pipeline(file_path)
            lg.info("Prediction completed successfully.")
            return jsonify({"prediction": prediction_result})

        except CustomException as e:
            lg.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return render_template("upload_file.html")

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
