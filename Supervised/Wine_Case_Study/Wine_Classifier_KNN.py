import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
# Date          : 07/03/2026
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
# Function Name : PerformEDA
# Description   : Performs exploratory data analysis.
# Parameters    : df (DataFrame)
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def PerformEDA(df):

    print("\nClass Distribution")
    print(df["Class"].value_counts())

    corr = df.corr()

    plt.figure(figsize=(12,8))
    sns.heatmap(corr,annot=True,cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTS_DIR/"CorrelationHeatmap.png")
    plt.close()

    sns.pairplot(df,hue="Class")
    plt.savefig(PLOTS_DIR/"PairPlot.png")
    plt.close()


# ==========================================================
# Function Name : DataSplit
# Description   : Splits the dataset into training and testing sets.
# Parameters    : df (DataFrame)
# Return        : X_train, X_test, Y_train, Y_test
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def DataSplit(df):

    X = df.drop("Class",axis=1)
    Y = df["Class"]

    return train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )


# ==========================================================
# Function Name : FeatureScaling
# Description   : Scales the feature data.
# Parameters    : X_train, X_test
# Return        : X_train_scaled, X_test_scaled, scaler
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def FeatureScaling(X_train,X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled,X_test_scaled,scaler


# ==========================================================
# Function Name : HyperParameterTuning
# Description   : Finds the best value of K.
# Parameters    : X_train, Y_train, X_test, Y_test
# Return        : best_k
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def HyperParameterTuning(X_train,Y_train,X_test,Y_test):

    accuracy_scores=[]

    k_values=range(1,21)

    for k in k_values:

        model=KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train,Y_train)

        prediction=model.predict(X_test)

        accuracy=accuracy_score(Y_test,prediction)

        accuracy_scores.append(accuracy)

    plt.figure(figsize=(8,5))
    plt.plot(k_values,accuracy_scores,marker="o")
    plt.grid(True)
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title("K vs Accuracy")
    plt.savefig(PLOTS_DIR/"K_vs_Accuracy.png")
    plt.close()

    best_k=list(k_values)[accuracy_scores.index(max(accuracy_scores))]

    print("\nBest K :",best_k)

    return best_k


# ==========================================================
# Function Name : ModelBuilding
# Description   : Builds and trains the KNN model.
# Parameters    : best_k, X_train, Y_train
# Return        : model
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def ModelBuilding(best_k,X_train,Y_train):

    model=KNeighborsClassifier(n_neighbors=best_k)

    model.fit(X_train,Y_train)

    return model


# ==========================================================
# Function Name : EvaluateModel
# Description   : Evaluates the trained model.
# Parameters    : model, X_test, Y_test
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def EvaluateModel(model,X_test,Y_test):

    prediction=model.predict(X_test)

    accuracy=accuracy_score(Y_test,prediction)

    print("\nAccuracy :",accuracy*100)

    cm=confusion_matrix(Y_test,prediction)

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(Y_test,prediction))

    display=ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display.plot()

    plt.savefig(PLOTS_DIR/"ConfusionMatrix.png")

    plt.close()


# ==========================================================
# Function Name : PreserveModel
# Description   : Saves the trained model and scaler.
# Parameters    : model, scaler
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def PreserveModel(model,scaler):

    timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    model_file=MODELS_DIR/f"WineKNN_{timestamp}.pkl"

    scaler_file=MODELS_DIR/f"WineScaler_{timestamp}.pkl"

    joblib.dump(model,model_file)

    joblib.dump(scaler,scaler_file)

    print("\nModel Saved :",model_file)

    print("Scaler Saved :",scaler_file)


# ==========================================================
# Function Name : main
# Description   : Controls the complete ML pipeline.
# Parameters    : None
# Return        : None
# Author        : Sakshi Ashok Adale
# Date          : 07/03/2026
# ==========================================================

def main():

    parser=argparse.ArgumentParser(
        description="Wine Classification using KNN"
    )

    parser.add_argument(
        "--data",
        default="WinePredictor.csv",
        help="Dataset Path"
    )

    args=parser.parse_args()

    print("*"*80)
    print("Wine Classification using KNN")
    print("*"*80)

    df=LoadDataset(args.data)

    PerformEDA(df)

    X_train,X_test,Y_train,Y_test=DataSplit(df)

    X_train_scaled,X_test_scaled,scaler=FeatureScaling(
        X_train,
        X_test
    )

    best_k=HyperParameterTuning(
        X_train_scaled,
        Y_train,
        X_test_scaled,
        Y_test
    )

    model=ModelBuilding(
        best_k,
        X_train_scaled,
        Y_train
    )

    EvaluateModel(
        model,
        X_test_scaled,
        Y_test
    )

    PreserveModel(
        model,
        scaler
    )

    print("\nPipeline Completed Successfully")
    print("Artifacts Stored in :",ARTIFACTS.resolve())


# ==========================================================
# Starter
# ==========================================================

if __name__=="__main__":
    main()