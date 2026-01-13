## Autism Spectrum Disorder Prediction using Machine Learning

An end-to-end machine learning project that predicts Autism Spectrum Disorder (ASD) using behavioral screening scores and demographic data. The project implements a complete ML pipeline including data preprocessing, exploratory data analysis, class imbalance handling, model training, hyperparameter tuning, and evaluation.

The best-performing model is selected using cross-validation and saved for deployment.



<!-- Problem Statement of the project -->
## Problem Statement

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects communication and behavior. Early screening can significantly improve diagnosis and intervention.

This project builds a machine learning system that predicts whether a person is likely to have ASD based on screening test responses and personal attributes.

## Project Features

- **Data cleaning and preprocessing**

- **Exploratory Data Analysis (EDA) with visualizations**

- **Handling missing values and outliers**

- **Label encoding of categorical features** 

- **Class imbalance handling using SMOTE**

- **Training multiple ML models:

    - Decision Tree

    - Random Forest

    - XGBoost

    - Hyperparameter tuning using RandomizedSearchCV

    - Model selection using cross-validation

    - Model evaluation using classification metrics

    - Model persistence for deployment

## Project Structure
.
├── data/
│   └── train.csv
├── main.ipynb
├── best_model.pkl
├── encoders.pkl
├── README.md

## Tech Stack

- **Python** 

- **Pandas, NumPy**

- **Matplotlib, Seaborn**

- **Scikit-learn**

- **Imbalanced-learn (SMOTE)**

- **XGBoost** 

## Machine Learning Pipeline
 
- **1. Data Loading & Understanding:-**

    -  **Dataset inspection**

    - **Data type validation**

    - **Missing value analysis**

- **2. Exploratory Data Analysis (EDA):-**

    - **Univariate analysis**

    - **Outlier detection using IQR**

    - **Correlation heatmap**

    - **Distribution analysis**

- **3. Data Preprocessing:-** 

    - **Missing value handling**

    - **Outlier treatment using median replacement**

    - **Label encoding of categorical features**

    - **Train-test split** 

- **4. Handling Class Imbalance**

    - **SMOTE oversampling on training dataset**

- **5. Model Training**

    - **Decision Tree**

    - **Random Forest**

    - **XGBoost**

- **6. Hyperparameter Tuning**

    - **RandomizedSearchCV for all models**

- **7. Model Selection** 

    - **Best model chosen based on cross-validation accuracy**

- **8. Model Evaluation**

    - **Accuracy Score**

    - **Confusion Matrix**

    - **Classification Report**

## Results

The best-performing model is evaluated on unseen test data using:

- **Accuracy**

- **Precision**

- **Recall**

- **F1-score**

The trained model is saved as best_model.pkl for future inference and deployment.

## How to Run the Project

- Step 1: Clone the repository 
    git clone <repository-url>
    cd autism-asd-prediction

- Step 2: Install dependencies
    pip install -r requirements.txt

- Step 3: Run the notebook
    jupyter notebook main.ipynb

## Future Enhancements

Build a web application for real-time prediction

Add explainable AI (SHAP/LIME)

Deploy using Flask or FastAPI

Add model monitoring and logging

## License

This project is intended for educational and research purposes.