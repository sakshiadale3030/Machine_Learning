import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================================
# Configuration
# ==========================================================

ARTIFACTS = Path("artifacts")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"

ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ==========================================================
# Function Name : LoadDataset
# Description   : Loads the dataset and displays basic details.
# Parameters    : filename (str)
# Return        : df (DataFrame)
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def LoadDataset(filename):
    df = pd.read_csv(filename)

    print(df.head())
    print("\nShape :", df.shape)
    print("\nDescription")
    print(df.describe())
    print("\nMissing Values")
    print(df.isnull().sum())

    return df


# ==========================================================
# Function Name : CleanTitanicData
# Description   : Preprocesses the dataset: drops unnecessary columns,
#                 handles missing values, and encodes categorical columns
#                 into numeric form.
# Parameters    : df (DataFrame)
# Return        : df (clean DataFrame)
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def CleanTitanicData(df):
    print("\nOriginal Data")
    print(df.head())

    # Remove unnecessary columns
    drop_columns = ["Passengerid", "zero", "Name", "Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\nColumns to be dropped :", existing_columns)
    df = df.drop(columns=existing_columns)

    # Handle Age column
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        age_median = df["Age"].median()
        df["Age"] = df["Age"].fillna(age_median)

    # Handle Fare column
    if "Fare" in df.columns:
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        fare_median = df["Fare"].median()
        df["Fare"] = df["Fare"].fillna(fare_median)

    # Handle Embarked column
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].astype(str).str.strip()
        df["Embarked"] = df["Embarked"].replace(["nan", "None", ""], np.nan)
        embarked_mode = df["Embarked"].mode()[0]
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    if "Sex" in df.columns:
        sex_as_text = df["Sex"].astype(str).str.strip().str.lower()

        if set(sex_as_text.unique()) <= {"male", "female"}:
            # Text format: map directly
            df["Sex"] = sex_as_text.map({"male": 0, "female": 1})
        else:
            # Already numeric (or numeric-like strings such as "0"/"1")
            df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")

        # Catch any unmapped/unexpected values instead of failing silently
        if df["Sex"].isnull().any():
            non_null_sex = df["Sex"].dropna()
            fill_value = non_null_sex.mode()[0] if not non_null_sex.empty else 0
            print(f"\nWarning: unexpected/missing values found in Sex column, "
                  f"filling with {fill_value}.")
            df["Sex"] = df["Sex"].fillna(fill_value)

    print("\nMissing values after preprocessing")
    print(df.isnull().sum())

    # Encode Embarked column
    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Convert boolean columns into integer (get_dummies produces bool columns)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nData after full preprocessing")
    print(df.head())
    print("Shape of dataset :", df.shape)

    return df


# ==========================================================
# Function Name : PerformEDA
# Description   : Performs exploratory data analysis and saves plots.
# Parameters    : df (DataFrame)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def PerformEDA(df):

    print("\nSurvival Distribution")
    print(df["Survived"].value_counts())

    plt.figure(figsize=(5, 4))
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Count")
    plt.savefig(PLOTS_DIR / "SurvivalCount.png")
    plt.close()

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "CorrelationHeatmap.png")
    plt.close()


# ==========================================================
# Function Name : DataSplit
# Description   : Splits the dataset into training and testing sets,
#                 preserving the survival ratio in both sets via stratify.
# Parameters    : df (DataFrame)
# Return        : X_train, X_test, Y_train, Y_test
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def DataSplit(df):

    X = df.drop("Survived", axis=1)
    Y = df["Survived"]

    print("\nShape of X :", X.shape)
    print("Shape of Y :", Y.shape)

    return train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )


# ==========================================================
# Function Name : FeatureScaling
# Description   : Scales the feature data. This matters for Logistic
#                 Regression since features like Fare and Age have much
#                 larger numeric ranges than binary/encoded columns, which
#                 would otherwise distort the model's coefficients and slow
#                 convergence.
# Parameters    : X_train, X_test
# Return        : X_train_scaled, X_test_scaled, scaler
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def FeatureScaling(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# ==========================================================
# Function Name : HyperParameterTuning
# Description   : Searches over regularization strength (C) values to find
#                 the one that gives the best test accuracy, and saves a
#                 plot of C vs Accuracy.
# Parameters    : X_train, Y_train, X_test, Y_test
# Return        : best_c (float)
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def HyperParameterTuning(X_train, Y_train, X_test, Y_test):

    c_values = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    accuracy_scores = []

    for c in c_values:
        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(Y_test, prediction)
        accuracy_scores.append(accuracy)

    plt.figure(figsize=(8, 5))
    plt.plot(c_values, accuracy_scores, marker="o")
    plt.xscale("log")
    plt.grid(True)
    plt.xlabel("C Value (log scale)")
    plt.ylabel("Accuracy")
    plt.title("C vs Accuracy")
    plt.savefig(PLOTS_DIR / "C_vs_Accuracy.png")
    plt.close()

    best_c = c_values[accuracy_scores.index(max(accuracy_scores))]
    print("\nBest C :", best_c)

    return best_c


# ==========================================================
# Function Name : ModelBuilding
# Description   : Builds and trains the Logistic Regression model using
#                 the best regularization strength found during tuning.
# Parameters    : best_c, X_train, Y_train
# Return        : model
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def ModelBuilding(best_c, X_train, Y_train):

    model = LogisticRegression(C=best_c, max_iter=1000)
    model.fit(X_train, Y_train)

    return model


# ==========================================================
# Function Name : EvaluateModel
# Description   : Evaluates the trained model: prints accuracy, confusion
#                 matrix, and classification report, and saves a confusion
#                 matrix plot.
# Parameters    : model, X_test, Y_test, feature_names
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def EvaluateModel(model, X_test, Y_test, feature_names):

    prediction = model.predict(X_test)

    accuracy = accuracy_score(Y_test, prediction)
    print("\nAccuracy :", accuracy * 100)

    # NOTE: sklearn convention is confusion_matrix(y_true, y_pred).
    # The original script had these swapped, which transposes the matrix.
    cm = confusion_matrix(Y_test, prediction)
    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(Y_test, prediction))

    print("\nCoefficients")
    for feature, coefficient in zip(feature_names, model.coef_[0]):
        print(feature, ":", coefficient)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )
    display.plot()
    plt.title("Confusion Matrix - Titanic Survival")
    plt.savefig(PLOTS_DIR / "ConfusionMatrix.png")
    plt.close()


# ==========================================================
# Function Name : PreserveModel
# Description   : Saves the trained model and scaler with a timestamped
#                 filename, so previous runs are never overwritten.
# Parameters    : model, scaler
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def PreserveModel(model, scaler):

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    model_file = MODELS_DIR / f"TitanicLogistic_{timestamp}.pkl"
    scaler_file = MODELS_DIR / f"TitanicScaler_{timestamp}.pkl"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    print("\nModel Saved :", model_file)
    print("Scaler Saved :", scaler_file)


# ==========================================================
# Function Name : LoadPreservedModel
# Description   : Loads a previously saved model and scaler from disk.
# Parameters    : model_file, scaler_file
# Return        : model, scaler
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def LoadPreservedModel(model_file, scaler_file):

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    print("Model and scaler successfully loaded")

    return model, scaler


# ==========================================================
# Function Name : TitanicLogistic
# Description   : Main pipeline controller. Loads the dataset, performs
#                 EDA, preprocesses data, splits/scales it, tunes
#                 hyperparameters, trains, evaluates, and saves the model.
# Parameters    : data_path (str)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def TitanicLogistic(data_path):

    print("*" * 80)
    print("Titanic Survival Prediction using Logistic Regression")
    print("*" * 80)

    df = LoadDataset(data_path)

    df = CleanTitanicData(df)

    PerformEDA(df)

    X_train, X_test, Y_train, Y_test = DataSplit(df)

    feature_names = X_train.columns.tolist()

    X_train_scaled, X_test_scaled, scaler = FeatureScaling(X_train, X_test)

    best_c = HyperParameterTuning(
        X_train_scaled, Y_train, X_test_scaled, Y_test
    )

    model = ModelBuilding(best_c, X_train_scaled, Y_train)

    EvaluateModel(model, X_test_scaled, Y_test, feature_names)

    PreserveModel(model, scaler)

    print("\nPipeline Completed Successfully")
    print("Artifacts stored in :", ARTIFACTS.resolve())


# ==========================================================
# Function Name : main
# Description   : Starting point of the application. Parses the dataset
#                 path from the command line.
# Parameters    : None
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 14/03/2026
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Titanic Survival Prediction using Logistic Regression"
    )

    parser.add_argument(
        "--data",
        default="TitanicDataset.csv",
        help="Dataset path"
    )

    args = parser.parse_args()

    TitanicLogistic(args.data)


if __name__ == "__main__":
    main()