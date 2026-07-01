import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from pathlib import Path
import joblib
import argparse
from datetime import datetime

# CONFIGURATION DETAILS
ARTIFACTS = Path("artifacts")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

###############################################################################################################
# Function Name :- load_dataset()
# Description   :- Loads the dataset, prints sample, shape, and summary.
# Parameters    :- filename (str) - Path to the CSV dataset.
# Return        :- DataFrame - Loaded dataset.
# Author        :- Sakshi Ashok Adale
# Date          :- 07/03/2026
###############################################################################################################

def load_dataset(filename):
    df = pd.read_csv(filename)

    # Drop extra index column if exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    print("Dataset sample:\n", df.head())
    print("\nShape:", df.shape)
    print("\nStatistical Summary:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())
    return df

###############################################################################################################
# Function Name :- correlationHeatmap()
# Description   :- Generates correlation heatmap and pairplot.
# Parameters    :- df (DataFrame) - Input dataset.
# Return        :- DataFrame - Correlation matrix.
# Author        :- Sakshi Ashok Adale
# Date          :- 07/03/2026
###############################################################################################################

def correlationHeatmap(df):
    corrmatrix = df.corr()

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corrmatrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    # Pairplot
    sns.pairplot(df)
    plt.suptitle("Pairplot of Independent Features", y=1.02)
    plt.savefig(PLOTS_DIR / "pairplot.png")
    plt.close()

    print("\nCorrelation Matrix:\n", corrmatrix)
    return corrmatrix

###############################################################################################################
# Function Name :- DataSplit()
# Description   :- Splits the dataset into training and testing sets.
# Parameters    :- df (DataFrame) - Input dataset.
# Return        :- X_train, X_test, Y_train, Y_test.
# Author        :- Sakshi Ashok Adale
# Date          :- 07/03/2026
###############################################################################################################

def DataSplit(df):
    X = df[['TV', 'radio', 'newspaper']]
    y = df[['sales']]
    return train_test_split(X, y, test_size=0.2, random_state=42)

###############################################################################################################
# Function Name :- ModelBuilding()
# Description   :- Builds, trains, and evaluates the Linear Regression model.
# Parameters    :- X_train (DataFrame), X_test (DataFrame),
#                  Y_train (DataFrame), Y_test (DataFrame)
# Return        :- Pipeline - Trained machine learning pipeline.
# Author        :- Sakshi Ashok Adale
# Date          :- 07/03/2026
###############################################################################################################

def ModelBuilding(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_test)

    mse = metrics.mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(Y_test, y_pred)

    print("\nModel Evaluation Metrics:")
    print("Mean Squared Error :", mse)
    print("Root Mean Squared Error :", rmse)
    print("R Square :", r2)

    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, y_pred, color="blue")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png")
    plt.close()

    # Residuals
    residuals = Y_test.values.flatten() - y_pred.flatten()

    # Residual plot (no statsmodels)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred.flatten(), residuals, alpha=0.7, color="red")
    plt.axhline(0, linestyle="--", linewidth=1, color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "residual_plot.png")
    plt.close()

    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=20, color="purple")
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig(PLOTS_DIR / "residual_distribution.png")
    plt.close()

    return pipeline

###############################################################################################################
# Function Name :- featureImportance()
# Description   :- Displays feature coefficients and their importance.
# Parameters    :- model (Pipeline), feature_names (Index/List)
# Return        :- Series - Feature importance values.
# Author        :- Sakshi Ashok Adale
# Date          :- 07/03/2026
###############################################################################################################

def featureImportance(model, feature_names):
    regressor = model.named_steps["regressor"]
    coefficients = regressor.coef_[0]
    intercept = regressor.intercept_

    importance = pd.Series(coefficients, index=feature_names).sort_values(key=abs, ascending=False)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Feature Importance (Linear Regression Coefficients)")
    plt.ylabel("Coefficient Value")
    plt.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close()

    print("\nModel Coefficients:")
    for col, coef in zip(feature_names, coefficients):
        print(f"{col}: {coef}")
    print("Intercept:", intercept)

    return importance

###############################################################################################################
# Function Name :- PreservetheModel()
# Description   :- Saves the trained machine learning model.
# Parameters    :- model (Pipeline) - Trained model.
# Return        :- Path - Saved model file path.
# Author        :- Sakshi Ashok Adale
# Date          :- 07/03/2026
###############################################################################################################

def PreservetheModel(model):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_file = MODELS_DIR / f"advertisement_model_{timestamp}.pkl"
    joblib.dump(model, model_file)
    print(f"\nModel saved at: {model_file}")
    return model_file

###############################################################################################################
# Function Name :- main()
# Description   :- Executes the complete advertisement sales prediction pipeline.
# Parameters    :- None
# Return        :- None
# Author        :- Sakshi Ashok Adale
# Date          :- 07/03/2026
###############################################################################################################

def main():
    parser = argparse.ArgumentParser(description="Advertisement Sales Predictor Application")
    parser.add_argument("--data", type=str, default="Advertising.csv", help="Path to dataset CSV")
    args = parser.parse_args()

    line = "*" * 84
    print(line)
    print("--------------------------- Advertisement Case Study Application ---------------------------")

    # Load dataset
    df = load_dataset(args.data)

    # EDA
    correlationHeatmap(df)

    # Split data
    X_train, X_test, Y_train, Y_test = DataSplit(df)

    # Model building + evaluation
    model = ModelBuilding(X_train, X_test, Y_train, Y_test)

    # Feature importance
    featureImportance(model, df.drop(columns=["sales"]).columns)

    # Save model
    PreservetheModel(model)

    print(line)
    print("Pipeline trained & saved successfully.")
    print(f"Artifacts stored in: {ARTIFACTS.resolve()}")
    print(line)

###############################################################################################################
## Starter
###############################################################################################################

if __name__ == "__main__":
    main()