# Customer Segmentation

## Overview

This project implements a **Customer Segmentation System** using **Unsupervised Machine Learning** techniques. It generates a synthetic customer dataset, preprocesses the data, determines the optimal number of clusters, performs customer segmentation using multiple clustering algorithms, visualizes the results, analyzes customer groups, and provides business recommendations for each customer segment.

---

## Features

* Generates synthetic customer data
* Data preprocessing and feature engineering
* Label encoding for categorical features
* Feature scaling using StandardScaler
* Finds the optimal number of clusters using:

  * Elbow Method
  * Silhouette Score
  * Calinski-Harabasz Score
* Implements multiple clustering algorithms:

  * K-Means Clustering
  * Hierarchical (Agglomerative) Clustering
* Selects the best clustering algorithm automatically
* Creates multiple visualization plots
* Performs detailed customer segment analysis
* Generates business recommendations for every customer segment
* Saves trained models, scaler, and label encoders for future use

---

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## Project Structure

```
Customer_Segmentation_Case_Study/
│
├── artifacts/
│   ├── models/
│   └── plots/
│
├── Customer_Segmentation.py
├── README.md
└── requirements.txt
```

---

## Dataset

The project generates a synthetic customer dataset containing features such as:

* Customer ID
* Age
* Annual Income
* Spending Score
* Gender
* Education Level
* Marital Status
* Number of Children
* Credit Score
* Years as Customer
* Total Purchases
* Average Order Value
* Days Since Last Purchase
* Product Category Preference
* Channel Preference

---

## Machine Learning Workflow

1. Generate customer dataset.
2. Preprocess numerical and categorical data.
3. Encode categorical features.
4. Scale numerical features.
5. Determine the optimal number of clusters.
6. Perform K-Means and Hierarchical Clustering.
7. Compare clustering performance.
8. Visualize customer segments.
9. Analyze each segment.
10. Generate business recommendations.
11. Save trained models and preprocessing artifacts.

---

## Output

The project produces:

* Customer segmentation plots
* Cluster evaluation graphs
* Segment analysis
* Business recommendations
* Saved clustering model
* Saved scaler
* Saved label encoders

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Customer_Segmentation.git
```

Move into the project directory:

```bash
cd Customer_Segmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Run using default settings:

```bash
python customer_segmentation.py
```

Run with custom parameters:

```bash
python customer_segmentation.py --customers 1500 --max_k 10
```

---

## Results

The system automatically:

* Finds the optimal number of customer segments.
* Compares K-Means and Hierarchical Clustering.
* Selects the best-performing model.
* Generates detailed visualizations.
* Profiles each customer segment.
* Suggests business strategies for marketing and customer retention.

---

## Future Improvements

* Support real-world customer datasets
* Interactive dashboard using Streamlit
* Customer prediction for new users
* Additional clustering algorithms (DBSCAN, Gaussian Mixture Models)
* Automated report generation

---

## Author

**Sakshi Ashok Adale**

Machine Learning | Python | Data Science

---

## License

This project is intended for educational and learning purposes.
