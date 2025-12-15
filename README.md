# Customer Churn Prediction System

## Overview
This project implements a machine learning system to predict customer churn using structured business data. The objective is to identify customers who are likely to discontinue a service, enabling proactive retention strategies. The project demonstrates an end-to-end ML workflow, from data preprocessing and feature engineering to model training, evaluation, and interpretation.


## Problem Statement
Customer churn is a critical challenge for subscription-based and service-driven businesses. Accurately predicting churn allows organisations to target at-risk customers, reduce revenue loss, and improve customer satisfaction. This project focuses on building a robust and interpretable churn prediction model using classical machine learning techniques.


## Solution Approach
The system follows a modular pipeline:
**1. Data Preprocessing**
     • Handles missing values
     • Encodes categorical variables
     • Produces a clean, model-ready dataset
**2. Model Training**
     • Trains multiple classification models
     • Compares performance using ROC-AUC
     • Selects and persists the best-performing model
**3. Model Evaluation**
     • Evaluates performance using precision, recall, F1-score, and ROC-AUC
     • Analyses confusion matrix to understand business impact

     
## Technologies Used
• Programming Language: Python
• Data Processing: Pandas, NumPy
• Machine Learning: Scikit-learn
• Model Evaluation: ROC-AUC, Classification Report
• Version Control: Git, GitHub


## Project Structure
customer-churn-prediction-system/
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── model/
│   └── churn_model.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md


## How to Run the Project
1. Install dependencies:
   pip install -r requirements.txt
2. Preprocess the data:
   python src/data_preprocessing.py
3. Train the model:
   python src/train_model.py
4. Evaluate the model:
   python src/evaluate_model.py


## Key Results
   • Successfully trained and evaluated churn prediction models
   • Used ROC-AUC to select the best-performing algorithm
   • Delivered interpretable metrics aligned with business decision-making


## Business Impact
   • Identifies high-risk customers before churn occurs
   • Enables targeted retention campaigns
   • Supports data-driven decision-making for customer success teams

   
## Future Enhancements
   • Advanced feature engineering
   • Hyperparameter tuning
   • Model explainability (e.g., SHAP)
   • Deployment as a REST API

   
## Author
### Nissiya Thomas
MSc Advanced Computer Science – University of Liverpool
