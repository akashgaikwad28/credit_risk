# Credit Risk Prediction Platform

![Banner Image](https://via.placeholder.com/1000x300?text=Credit+Risk+Prediction+Platform)

## Overview

The **Credit Risk Prediction Platform** is a machine learning-powered web application designed to assess the creditworthiness of individuals based on user-provided financial data. The project is hosted on Render and available at: [Credit Risk Platform](https://credit-risk-cc62.onrender.com/).

This platform predicts whether a loan applicant is at "low" or "high" risk of defaulting on their loan. The tool aims to aid financial institutions in making data-driven decisions.

---

## Features

- **Upload Financial Data**: Accepts CSV files with applicant financial information.
- **Real-time Predictions**: Processes data and provides a credit risk assessment instantly.
- **Scalable Deployment**: Hosted on Render for seamless user access.
- **User-Friendly Interface**: Simplified and intuitive design for all stakeholders.

---

## How It Works

1. **Upload Data**: The user uploads a CSV file containing applicant details.
2. **Preprocessing**: Data undergoes cleaning and transformation using a pre-built pipeline.
3. **Prediction**: The model predicts credit risk as "Good" or "Bad".
4. **Results**: Results are displayed in tabular form and made available for download.

---

## Tech Stack

### Frontend:
- **HTML**, **CSS**

### Backend:
- **Flask**: Powers the web application.

### Machine Learning:
- **Scikit-learn**: Used to develop the classification model.
- **Pandas**: For data manipulation.
- **Pickle**: For model serialization.

### Deployment:
- **Render**: Used for deploying the web application.
- **Docker**: Containerization for a consistent environment.

---

## Dataset

The platform uses a dataset containing:
- Applicant demographic data
- Employment history
- Credit history
- Financial information

Target variable: **Credit Risk** (Good/Bad).

Example dataset:

| ID   | Age | Income | Loan Amount | Credit History | Employment Status | Credit Risk |
|------|-----|--------|-------------|----------------|-------------------|-------------|
| 101  | 35  | 50000  | 20000       | 1              | Employed          | Good        |
| 102  | 29  | 30000  | 15000       | 0              | Unemployed        | Bad         |

---

## Model Pipeline

1. **Data Preprocessing**
   - Handles missing values.
   - Encodes categorical variables.
   - Scales numerical features.

2. **Model Training**
   - Algorithm: Random Forest Classifier.
   - Metrics: Accuracy, F1-score, and ROC-AUC.

3. **Prediction Pipeline**
   - Automated prediction using the serialized model and preprocessor.

---

## Visual Representation

### Workflow Diagram

![Workflow](https://via.placeholder.com/800x400?text=Workflow+Diagram)

### Prediction Example

**Input Data:**

| Age | Income | Loan Amount | Credit History | Employment Status |
|-----|--------|-------------|----------------|-------------------|
| 35  | 50000  | 20000       | 1              | Employed          |

**Output:**

| Age | Income | Loan Amount | Credit History | Employment Status | Credit Risk |
|-----|--------|-------------|----------------|-------------------|-------------|
| 35  | 50000  | 20000       | 1              | Employed          | Good        |

### Illustrative Icons

![Credit Card Icon](https://via.placeholder.com/150?text=Credit+Card)
![Bank Icon](https://via.placeholder.com/150?text=Bank)

---

## Getting Started

### Prerequisites

- Python 3.8+
- Required Libraries:
  - Flask
  - Pandas
  - Scikit-learn
  - dotenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/akashgaikwad28/credit_risk.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit_risk
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file and define `TARGET_COLUMN` and `ARTIFACT_FOLDER`.

5. Run the application:
   ```bash
   python app.py
   ```

---

## Deployment

The project is deployed on Render for public access. For deployment on your own infrastructure, use the provided Dockerfile:

1. Build the Docker image:
   ```bash
   docker build -t credit-risk-prediction .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 credit-risk-prediction
   ```

---

## Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---


## Contact

For any questions, reach out to:
- **Name**: Akash Gaikwad
- **Email**: acashtech28@gmail.com
- **GitHub**: [akashgaikwad28](https://github.com/akashgaikwad28)

---

![Thank You](https://via.placeholder.com/1000x200?text=Thank+You+for+Exploring+the+Project)

