import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from pathlib import Path
import joblib
import argparse
from datetime import datetime

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
# Date          : 21/02/2026
# ==========================================================

def LoadDataset(filename):

    df = pd.read_csv(filename)

    print(df.head())
    print(df.shape)
    print(df.describe())
    print(df.isnull().sum())

    return df

# ==========================================================
# Function Name : PerformEDA
# Description   : Performs exploratory data analysis.
# Parameters    : df (DataFrame)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def PerformEDA(df):

    print(df["species"].value_counts())

    corr = df.drop("species",axis=1).corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr,annot=True,cmap="coolwarm")

    plt.title("Correlation Heatmap")

    plt.savefig(PLOTS_DIR/"CorrelationHeatmap.png")

    plt.close()

    sns.pairplot(df,hue="species")

    plt.savefig(PLOTS_DIR/"PairPlot.png")

    plt.close()

# ==========================================================
# Function Name : ScatterPlot
# Description   : Creates and saves a scatter plot of petal length vs petal width.
# Parameters    : df (DataFrame) - Input dataset.
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def ScatterPlot(df):

    plt.figure(figsize=(7,5))

    for sp in df["species"].unique():

        temp=df[df["species"]==sp]

        plt.scatter(
            temp["petal length (cm)"],
            temp["petal width (cm)"],
            label=sp
        )

    plt.legend()

    plt.grid(True)

    plt.xlabel("Petal Length")

    plt.ylabel("Petal Width")

    plt.savefig(PLOTS_DIR/"ScatterPlot.png")

    plt.close()

# ==========================================================
# Function Name : DataSplit
# Description   : Splits the dataset into training and testing sets.
# Parameters    : df (DataFrame) - Input dataset.
# Return        : X_train, X_test, Y_train, Y_test
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def DataSplit(df):

    X=df.drop("species",axis=1)

    Y=df["species"]

    return train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

# ==========================================================
# Function Name : BuildModel
# Description   : Builds and trains the Decision Tree classifier.
# Parameters    : X_train (DataFrame), Y_train (Series)
# Return        : model (DecisionTreeClassifier)
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def BuildModel(X_train,Y_train):

    model=DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )

    model.fit(X_train,Y_train)

    return model

# ==========================================================
# Function Name : EvaluateModel
# Description   : Evaluates the trained model using the test dataset.
# Parameters    : model (DecisionTreeClassifier),
#                 X_test (DataFrame),
#                 Y_test (Series)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def EvaluateModel(model,X_test,Y_test):

    Y_pred=model.predict(X_test)

    accuracy=accuracy_score(Y_test,Y_pred)

    print("Accuracy :",accuracy*100)

    cm=confusion_matrix(Y_test,Y_pred)

    print(cm)

    print(classification_report(Y_test,Y_pred))

    display=ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display.plot()

    plt.savefig(PLOTS_DIR/"ConfusionMatrix.png")

    plt.close()

# ==========================================================
# Function Name : PlotDecisionTree
# Description   : Plots and saves the trained Decision Tree.
# Parameters    : model (DecisionTreeClassifier)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def PlotDecisionTree(model):

    plt.figure(figsize=(18,10))

    plot_tree(
        model,
        filled=True,
        feature_names=[
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],
        class_names=model.classes_
    )

    plt.savefig(PLOTS_DIR/"DecisionTree.png")

    plt.close()

# ==========================================================
# Function Name : FeatureImportance
# Description   : Displays and plots feature importance scores.
# Parameters    : model (DecisionTreeClassifier),
#                 features (Index/List)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def FeatureImportance(model,features):

    importance=pd.Series(
        model.feature_importances_,
        index=features
    )

    importance=importance.sort_values(ascending=False)

    print(importance)

    plt.figure(figsize=(7,5))

    importance.plot(kind="bar")

    plt.title("Feature Importance")

    plt.savefig(PLOTS_DIR/"FeatureImportance.png")

    plt.close()

# ==========================================================
# Function Name : PreserveModel
# Description   : Saves the trained model with a timestamp.
# Parameters    : model (DecisionTreeClassifier)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def PreserveModel(model):

    timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    filename=MODELS_DIR/f"IrisDecisionTree_{timestamp}.pkl"

    joblib.dump(model,filename)

    print("Model saved :",filename)

# ==========================================================
# Function Name : main
# Description   : Executes the complete Decision Tree classification pipeline.
# Parameters    : None
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 21/02/2026
# ==========================================================

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="iris.csv"
    )

    args=parser.parse_args()

    df=LoadDataset(args.data)

    PerformEDA(df)

    ScatterPlot(df)

    X_train,X_test,Y_train,Y_test=DataSplit(df)

    model=BuildModel(X_train,Y_train)

    EvaluateModel(model,X_test,Y_test)

    PlotDecisionTree(model)

    FeatureImportance(
        model,
        df.drop("species",axis=1).columns
    )

    PreserveModel(model)

    print("Pipeline Completed Successfully")      

# ==========================================================
# Program entry point
# ==========================================================

if __name__ == "__main__":
    main()                      