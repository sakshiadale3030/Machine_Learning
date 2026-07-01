# 🍷 Wine Classification using K-Nearest Neighbors (KNN)

## 📖 Project Overview

This project implements a **Wine Classification System** using the **K-Nearest Neighbors (KNN)** Machine Learning algorithm. The objective is to classify different types of wine based on their chemical properties.

The project follows a complete Machine Learning pipeline, including:

- Dataset Loading
- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Pair Plot Visualization
- Data Splitting
- Feature Scaling
- Hyperparameter Tuning (Finding Best K)
- KNN Model Training
- Model Evaluation
- Confusion Matrix Visualization
- Model Persistence using Joblib

All generated plots and trained models are automatically stored inside the **artifacts** directory.

---

# 📂 Dataset

Dataset Name:

**WinePredictor.csv**

The dataset contains several chemical attributes of wines along with their class labels.

## Input Features

Some common features include:

- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315
- Proline

## Target Variable

- Class

---

# 🛠 Technologies Used

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Pathlib
- Argparse

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Wine-KNN-Classifier.git
cd Wine-KNN-Classifier
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# 📁 Project Structure

```
Wine-KNN-Classifier/
│
├── WinePredictor.csv
├── Wine_Classifier_KNN.py
├── README.md
├── requirements.txt
│
└── artifacts/
    ├── models/
    │      ├── WineKNN_YYYYMMDD_HHMMSS.pkl
    │      └── WineScaler_YYYYMMDD_HHMMSS.pkl
    │
    └── plots/
           ├── CorrelationHeatmap.png
           ├── PairPlot.png
           ├── K_vs_Accuracy.png
           └── ConfusionMatrix.png
```

---

# 🚀 How to Run

Using the default dataset

```bash
python Wine_Classifier_KNN.py
```

Using a custom dataset

```bash
python Wine_Classifier_KNN.py --data WinePredictor.csv
```

---

# ⚙️ Machine Learning Pipeline

## Step 1 : Load Dataset

Loads the dataset and displays:

- First five records
- Dataset shape
- Statistical summary
- Missing values

Function

```python
LoadDataset()
```

---

## Step 2 : Exploratory Data Analysis

Performs

- Correlation Matrix
- Correlation Heatmap
- Pair Plot
- Class Distribution

Generated Files

- CorrelationHeatmap.png
- PairPlot.png

Function

```python
PerformEDA()
```

---

## Step 3 : Data Splitting

Splits the dataset into

- Training Data (80%)
- Testing Data (20%)

using

```python
train_test_split()
```

Function

```python
DataSplit()
```

---

## Step 4 : Feature Scaling

Normalizes all input features using

- StandardScaler

Function

```python
FeatureScaling()
```

---

## Step 5 : Hyperparameter Tuning

Evaluates values of **K = 1 to 20**.

Plots

- K vs Accuracy Graph

Selects the best value of K based on maximum accuracy.

Function

```python
HyperParameterTuning()
```

---

## Step 6 : Model Building

Builds the final KNN model using the optimal K value.

Function

```python
ModelBuilding()
```

---

## Step 7 : Model Evaluation

Evaluates the model using

- Accuracy Score
- Confusion Matrix
- Classification Report

Generates

- ConfusionMatrix.png

Function

```python
EvaluateModel()
```

---

## Step 8 : Model Persistence

Saves the trained KNN model and StandardScaler using Joblib.

Example Files

```
WineKNN_20260701_113500.pkl

WineScaler_20260701_113500.pkl
```

Function

```python
PreserveModel()
```

---

# 📊 Generated Artifacts

```
artifacts/
│
├── models/
│      WineKNN_YYYYMMDD_HHMMSS.pkl
│      WineScaler_YYYYMMDD_HHMMSS.pkl
│
└── plots/
       CorrelationHeatmap.png
       PairPlot.png
       K_vs_Accuracy.png
       ConfusionMatrix.png
```

---

# 📈 Evaluation Metrics

The project evaluates the model using

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

# 🎯 Functions Implemented

| Function | Description |
|----------|-------------|
| `LoadDataset()` | Loads the dataset and displays information |
| `PerformEDA()` | Performs correlation analysis and visualization |
| `DataSplit()` | Splits the dataset into training and testing sets |
| `FeatureScaling()` | Scales input features using StandardScaler |
| `HyperParameterTuning()` | Finds the optimal K value |
| `ModelBuilding()` | Builds and trains the KNN model |
| `EvaluateModel()` | Evaluates the trained model |
| `PreserveModel()` | Saves the model and scaler |
| `main()` | Controls the complete machine learning pipeline |

---

# 🎯 Learning Outcomes

This project demonstrates

- Data Preprocessing
- Exploratory Data Analysis
- Correlation Analysis
- Feature Scaling
- K-Nearest Neighbors (KNN)
- Hyperparameter Tuning
- Model Evaluation
- Confusion Matrix
- Model Persistence
- Machine Learning Pipeline

---

# 🔮 Future Enhancements

- GridSearchCV Hyperparameter Tuning
- Cross Validation
- PCA for Dimensionality Reduction
- Feature Selection
- Streamlit Web Application
- Flask/FastAPI Deployment
- Prediction using User Input

---

# 👩‍💻 Author

**Sakshi Ashok Adale**

Date - **31/06/2026**
