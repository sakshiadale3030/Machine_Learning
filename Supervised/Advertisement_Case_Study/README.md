# 📺 Advertisement Sales Prediction using Linear Regression

## 📖 Project Overview

This project predicts product **sales** based on advertisement expenditure on different media platforms using **Linear Regression**.

The implementation follows a complete Machine Learning pipeline, including:

- Dataset Loading
- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Pair Plot Visualization
- Data Splitting
- Feature Scaling
- Linear Regression Model Building
- Model Evaluation
- Feature Importance Analysis
- Model Persistence using Joblib

All generated plots and trained models are automatically stored inside the **artifacts** directory.

---

# 📂 Dataset

Dataset Name:

**Advertising.csv**

The dataset contains advertising budgets spent on different media and the corresponding sales.

## Features

- TV Advertising Budget
- Radio Advertising Budget
- Newspaper Advertising Budget

## Target Variable

- Sales

---

# 🛠 Technologies Used

- Python 3.x
- Pandas
- NumPy
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
git clone https://github.com/yourusername/Advertisement-Sales-Prediction.git
cd Advertisement-Sales-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Project Structure

```
Advertisement-Sales-Prediction/
│
├── Advertising.csv
├── AdvertismentLinearRegression.py
├── README.md
├── requirements.txt
│
└── artifacts/
    ├── models/
    │     └── advertisement_model_YYYYMMDD_HHMMSS.pkl
    │
    └── plots/
          ├── correlation_heatmap.png
          ├── pairplot.png
          ├── actual_vs_predicted.png
          ├── residual_plot.png
          ├── residual_distribution.png
          └── feature_importance.png
```

---

# 🚀 How to Run

Run using the default dataset:

```bash
python AdvertisementPrediction.py
```

Run using a custom dataset:

```bash
python AdvertisementPrediction.py --data Advertising.csv
```

---

# ⚙️ Machine Learning Pipeline

## Step 1 : Load Dataset

The dataset is loaded using Pandas.

Displays:

- First five records
- Dataset shape
- Statistical summary
- Missing values

Function:

```python
load_dataset()
```

---

## Step 2 : Exploratory Data Analysis

Performs

- Correlation Matrix
- Correlation Heatmap
- Pair Plot

Generated Files

- correlation_heatmap.png
- pairplot.png

Function:

```python
correlationHeatmap()
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

Function:

```python
DataSplit()
```

---

## Step 4 : Model Building

Creates a Machine Learning Pipeline consisting of

- StandardScaler
- LinearRegression

Function:

```python
ModelBuilding()
```

---

## Step 5 : Model Evaluation

The trained model is evaluated using

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Generated Graphs

- Actual vs Predicted
- Residual Plot
- Residual Distribution

---

## Step 6 : Feature Importance

Displays Linear Regression coefficients for

- TV
- Radio
- Newspaper

Also generates

- Feature Importance Bar Graph

Function:

```python
featureImportance()
```

---

## Step 7 : Model Persistence

The trained model is saved using Joblib.

Example filename

```
advertisement_model_20260701_104520.pkl
```

Function:

```python
PreservetheModel()
```

---

# 📊 Generated Artifacts

```
artifacts/
│
├── models/
│     advertisement_model_YYYYMMDD_HHMMSS.pkl
│
└── plots/
      correlation_heatmap.png
      pairplot.png
      actual_vs_predicted.png
      residual_plot.png
      residual_distribution.png
      feature_importance.png
```

---

# 📈 Evaluation Metrics

The project reports

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 🎯 Functions Implemented

| Function | Description |
|----------|-------------|
| `load_dataset()` | Loads the dataset and displays dataset information |
| `correlationHeatmap()` | Generates correlation heatmap and pairplot |
| `DataSplit()` | Splits the dataset into training and testing sets |
| `ModelBuilding()` | Builds, trains and evaluates the Linear Regression model |
| `featureImportance()` | Displays feature coefficients and plots feature importance |
| `PreservetheModel()` | Saves the trained model using Joblib |
| `main()` | Controls the complete Machine Learning pipeline |

---

# 🎯 Learning Outcomes

This project demonstrates

- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Feature Scaling
- Linear Regression
- Model Evaluation
- Residual Analysis
- Feature Importance
- Model Persistence
- Machine Learning Pipeline
- Organized Project Structure

---

# 🔮 Future Enhancements

- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Hyperparameter Tuning
- Cross Validation
- Streamlit Dashboard
- Flask/FastAPI Deployment
- Prediction using User Input

---

# 👩‍💻 Author

**Sakshi Ashok Adale**

Date - **30/06/2026**