# Titanic Survival Prediction using Logistic Regression

A Machine Learning classification project that predicts whether a passenger survived the Titanic disaster using Logistic Regression. The project follows a complete end-to-end ML pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and model persistence.

---

## Features

- Data Cleaning and Preprocessing
- Missing Value Handling
- Categorical Feature Encoding
- Feature Scaling
- Exploratory Data Analysis (EDA)
- Logistic Regression Model
- Model Evaluation
- Confusion Matrix
- Classification Report
- Model Serialization using Joblib
- Command Line Execution using argparse
- Automatic Artifact Generation

---

## Project Structure

```
Titanic_Case_Study/
│
├── artifacts/
│   ├── models/
│   │   ├── TitanicLogistic.pkl
│   │   └── TitanicScaler.pkl
│   │
│   ├── plots/
│   │   ├── CorrelationHeatmap.png
│   │   ├── ConfusionMatrix.png
│   │   ├── SurvivalDistribution.png
│   │   └── AgeDistribution.png
│   │
│   └── reports/
│       └── ClassificationReport.txt
│
├── TitanicDataset.csv
│
├── Titanic_Logistic_Regression.py
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

1. Load Dataset
2. Perform Exploratory Data Analysis
3. Data Cleaning
4. Handle Missing Values
5. Encode Categorical Features
6. Feature Scaling
7. Split Dataset
8. Train Logistic Regression Model
9. Evaluate Model
10. Save Model and Scaler

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Installation

Go to the project directory

```bash
cd Titanic_Case_Study
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

Using the default dataset

```bash
python Titanic_Logistic_Regression.py
```

Using a custom dataset

```bash
python Titanic_Logistic_Regression.py --data TitanicDataset.csv
```

---

## Model

Algorithm Used

- Logistic Regression

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Artifacts Generated

### Models

- TitanicLogistic.pkl
- TitanicScaler.pkl

### Plots

- Correlation Heatmap
- Confusion Matrix
- Age Distribution
- Survival Distribution

### Reports

- Classification Report

---

## Future Improvements

- Hyperparameter Tuning
- Cross Validation
- ROC Curve
- AUC Score
- Flask/Streamlit Deployment
- Docker Support

---

## Author

**Sakshi Ashok Adale**

Date - **05/07/2026**

---