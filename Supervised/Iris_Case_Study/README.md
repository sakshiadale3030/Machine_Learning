# 🌸 Iris Flower Classification using Decision Tree

## 📖 Project Overview

This project implements a **Decision Tree Classification** model to classify Iris flowers into one of three species based on their sepal and petal measurements.

The project follows a complete Machine Learning pipeline, including:

- Dataset Loading
- Exploratory Data Analysis (EDA)
- Data Visualization
- Train-Test Split
- Decision Tree Model Training
- Model Evaluation
- Feature Importance Analysis
- Decision Tree Visualization
- Model Persistence using Joblib

All generated models and plots are automatically stored inside the **artifacts** folder.

---

# 📂 Dataset

Dataset Name:

**iris.csv**

The dataset contains **150 samples** with **4 numerical features** and **1 target variable**.

## Input Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Target Feature

- Species

Possible classes:

- Setosa
- Versicolor
- Virginica

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

# 📦 Required Libraries

Install all dependencies using:

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib
```

---

# 📁 Project Structure

```
IrisDecisionTree/
│
├── iris.csv
├── IrisDecisionTree.py
├── README.md
├── requirements.txt
│
└── artifacts/
    ├── models/
    │     └── IrisDecisionTree_YYYY_MM_DD_HH_MM_SS.pkl
    │
    └── plots/
          ├── CorrelationHeatmap.png
          ├── PairPlot.png
          ├── ScatterPlot.png
          ├── ConfusionMatrix.png
          ├── DecisionTree.png
          └── FeatureImportance.png
```

---

# 🚀 How to Run

Run using the default dataset:

```bash
python IrisDecisionTree.py
```

Run with a custom dataset:

```bash
python IrisDecisionTree.py --data iris.csv
```

---

# ⚙️ Project Workflow

## Step 1 – Load Dataset

The dataset is loaded using **Pandas**.

Displays:

- First 5 records
- Dataset Shape
- Statistical Summary
- Missing Values

Function Used:

```python
LoadDataset()
```

---

## Step 2 – Exploratory Data Analysis (EDA)

Performs:

- Species Distribution
- Correlation Heatmap
- Pair Plot

Generated Files

```
CorrelationHeatmap.png
PairPlot.png
```

Function Used:

```python
PerformEDA()
```

---

## Step 3 – Scatter Plot Visualization

Visualizes relationship between:

- Petal Length
- Petal Width

Generated File

```
ScatterPlot.png
```

Function Used:

```python
ScatterPlot()
```

---

## Step 4 – Data Splitting

The dataset is divided into:

- Training Data (80%)
- Testing Data (20%)

using

```python
train_test_split()
```

Function Used

```python
DataSplit()
```

---

## Step 5 – Model Building

The model used is

```python
DecisionTreeClassifier
```

Parameters

```python
criterion="gini"

max_depth=5

random_state=42
```

Function Used

```python
BuildModel()
```

---

## Step 6 – Model Evaluation

Performance metrics include:

- Accuracy Score
- Confusion Matrix
- Classification Report

Generated File

```
ConfusionMatrix.png
```

Function Used

```python
EvaluateModel()
```

---

## Step 7 – Decision Tree Visualization

Displays the complete trained Decision Tree.

Generated File

```
DecisionTree.png
```

Function Used

```python
PlotDecisionTree()
```

---

## Step 8 – Feature Importance

Displays the importance score of every feature.

Generated File

```
FeatureImportance.png
```

Function Used

```python
FeatureImportance()
```

---

## Step 9 – Model Persistence

The trained model is saved automatically using **Joblib**.

Example

```
IrisDecisionTree_2026_06_30_16_25_18.pkl
```

Function Used

```python
PreserveModel()
```

---

# 📊 Generated Artifacts

After successful execution the following files are created automatically.

```
artifacts/
│
├── models/
│     IrisDecisionTree_YYYY_MM_DD_HH_MM_SS.pkl
│
└── plots/
      CorrelationHeatmap.png
      PairPlot.png
      ScatterPlot.png
      ConfusionMatrix.png
      DecisionTree.png
      FeatureImportance.png
```

---

# 📈 Model Evaluation

The application reports:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Support

---

# 🎯 Functions Implemented

| Function | Description |
|----------|-------------|
| `LoadDataset()` | Loads the dataset and displays dataset information |
| `PerformEDA()` | Performs exploratory data analysis and saves plots |
| `ScatterPlot()` | Generates scatter plot of petal measurements |
| `DataSplit()` | Splits dataset into training and testing sets |
| `BuildModel()` | Creates and trains the Decision Tree model |
| `EvaluateModel()` | Evaluates the trained model |
| `PlotDecisionTree()` | Saves graphical representation of the Decision Tree |
| `FeatureImportance()` | Displays and saves feature importance chart |
| `PreserveModel()` | Saves the trained model using Joblib |
| `main()` | Executes the complete machine learning pipeline |

---

# 🎯 Learning Outcomes

This project demonstrates:

- Exploratory Data Analysis
- Data Visualization
- Feature Selection
- Classification using Decision Tree
- Model Evaluation
- Feature Importance
- Decision Tree Visualization
- Model Serialization
- Organized Project Structure
- Artifact Management

---

# 🔮 Future Enhancements

- Hyperparameter Tuning
- Cross Validation
- Grid Search CV
- Random Forest Comparison
- Streamlit Web Application
- User Prediction Interface
- REST API using Flask or FastAPI
- Deployment on Cloud

---

# 👩‍💻 Author

**Sakshi Ashok Adale**

Date - **30/06/2026**

---