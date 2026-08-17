# Loan Approval Prediction System

A machine learning web application that predicts whether a loan application is likely to be approved based on applicant, income, loan, credit history, and property information. The project covers the full workflow from data exploration and preprocessing to model training, evaluation, serialization, and deployment through a Flask web interface.

## Overview

This project uses supervised classification to estimate loan approval outcomes. Applicant details are entered through a responsive web form, transformed with the saved preprocessing scaler, and passed to a trained machine learning model. The application returns the predicted loan status along with the approval probability.

The repository also includes a Jupyter notebook for experimentation, a dataset, serialized model files, and generated evaluation visuals such as feature importance, confusion matrix, and ROC curve plots.

## Key Features

- Loan approval prediction from applicant and loan details
- Flask-based web application
- Responsive Bootstrap user interface
- Data cleaning and preprocessing workflow
- Feature encoding and scaling
- Model training and evaluation notebook
- Saved model and scaler using Joblib
- Approval probability output
- Model performance dashboard with evaluation charts

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- HTML
- CSS
- Bootstrap 5

## Machine Learning Workflow

1. Load loan applicant dataset
2. Clean and preprocess data
3. Encode categorical features
4. Scale numerical features
5. Split data for training and testing
6. Train classification models
7. Evaluate model performance
8. Save the trained model and scaler
9. Serve predictions through Flask

## Input Features

The prediction form uses the following fields:

- Gender
- Married status
- Number of dependents
- Education
- Self-employment status
- Applicant income
- Coapplicant income
- Loan amount
- Loan amount term
- Credit history
- Property area

## Project Structure

```text
Loan-Approval-Prediction-System/
+-- app/
|   +-- app.py
|   +-- static/
|   |   +-- confusion_matrix.png
|   |   +-- feature_importance.png
|   |   +-- roc_curve.png
|   |   +-- style.css
|   +-- templates/
|       +-- index.html
+-- data/
|   +-- loan_data.csv
+-- models/
|   +-- loan_model.pkl
|   +-- scaler.pkl
+-- notebooks/
|   +-- Loan_Approval.ipynb
+-- screenshots/
|   +-- feature_importance.png
|   +-- roc_curve.png
+-- README.md
+-- requirements.txt
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Loan-Approval-Prediction-System.git
cd Loan-Approval-Prediction-System
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask app

```bash
cd app
python app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Model Evaluation

The project includes evaluation assets for reviewing model behavior:

- Confusion matrix
- ROC curve
- Feature importance chart

These visuals are available in the `app/static/` and `screenshots/` directories.

## Repository Description

Use this short description for the GitHub repository "About" section:

```text
A Flask-based machine learning web app that predicts loan approval status using applicant, income, credit, and property details, with model evaluation visuals and a saved Scikit-learn pipeline.
```

## Future Improvements

- Add REST API endpoints for external integrations
- Support bulk CSV prediction
- Store prediction history in a database
- Add user authentication
- Containerize the app with Docker
- Deploy the application to a cloud platform

## Author

Sarfaraj Alam

B.Tech CSE (Data Science)  
Meerut Institute of Technology

## License

This project is developed for educational and internship purposes.
