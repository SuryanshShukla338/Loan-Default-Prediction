Loan Default Prediction 🚀

A machine learning project to predict loan defaults using Logistic Regression and Random Forest, with a complete preprocessing and evaluation pipeline.

📌 Project Overview

This project builds and compares classification models to predict whether a loan applicant will default. It demonstrates end-to-end ML workflow, from data preprocessing to model training, hyperparameter tuning, and evaluation.

⚙️ Key Features

Data Preprocessing:

Cleaned and standardized datasets.

Encoded categorical features with OneHotEncoder and scaled numerical variables with StandardScaler using ColumnTransformer.

Model Development:

Implemented Logistic Regression and Random Forest classifiers.

Applied GridSearchCV for hyperparameter tuning.

Model Evaluation:

Measured Accuracy, F1-score, ROC-AUC, and generated confusion matrices.

Visualized results using Matplotlib and Seaborn.

Deployment-Ready Pipeline:

Trained models consistently on train/test sets.

Automated predictions on unseen test data and exported results.

🛠️ Tech Stack

Languages & Libraries: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Methods: Classification, Feature Engineering, Hyperparameter Tuning, Model Evaluation

📊 Results

Improved loan default prediction accuracy by 15% over baseline.

Automated prediction pipeline generated results for 5K+ loan records.

🚀 How to Run

Clone this repository.

Place train.csv and test.csv in the project directory.

Run the script:

python loan_default_prediction.py


Predictions will be saved in outputs/test_predictions.csv and confusion matrix in outputs/confusion_matrix.png.

Suryansh Shukla
Aspiring Data & Research Analyst
