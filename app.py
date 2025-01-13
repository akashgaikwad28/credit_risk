from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
import os

# Initialize Flask app
app = Flask(__name__)

# Define the artifact folder path
artifact_folder = os.path.join(os.getcwd(), "artifacts")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/train", methods=["POST"])
def train_route():
    try:
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
        try:
            if 'file' not in request.files:
                lg.error("No file part in the request")
                return jsonify({"error": "No file part"}), 400
            
            file = request.files['file']
            if file.filename == '':
                lg.error("No selected file")
                return jsonify({"error": "No selected file"}), 400
            
            if not file.filename.endswith('.csv'):
                lg.error("Invalid file type")
                return jsonify({"error": "Only CSV files are allowed"}), 400
            
            # Save uploaded file
            file_path = os.path.join(artifact_folder, file.filename)
            

            # Run prediction pipeline
            prediction_pipeline = PredictionPipeline(request)
            prediction_config = prediction_pipeline.run_pipeline()
            
            
            file.save(file_path)
            lg.info("Input File saved successfully at artifacts.")
            lg.info("Prediction completed successfully.")
            
            # Get the prediction file path
            prediction_file_path = prediction_config.prediction_file_path
            
            # Check if file exists
            if not os.path.exists(prediction_file_path):
                lg.error("Prediction file not found")
                return jsonify({"error": "Prediction file not found"}), 500
            
            # Return the file as a download
            try:
                return send_file(
                    prediction_file_path,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='prediction_results.csv'
                )
            except Exception as e:
                lg.error(f"Error sending file: {str(e)}")
                return jsonify({"error": "Error downloading prediction file"}), 500

        except CustomException as e:
            lg.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    return render_template("upload_file.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)