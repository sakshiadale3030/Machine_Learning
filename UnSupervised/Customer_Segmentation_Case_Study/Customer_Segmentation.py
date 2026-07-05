import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
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
# Function name :- generate_customer_data()
# Description :- Generates synthetic customer dataset for segmentation
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def generate_customer_data(n_customers=1000, random_state=42):
    """Generate synthetic customer dataset for segmentation"""
    np.random.seed(random_state)
    
    # Generate customer data
    data = {
        'CustomerID': range(1, n_customers + 1),
        'Age': np.random.normal(35, 12, n_customers).astype(int),
        'Annual_Income': np.random.normal(50000, 20000, n_customers),
        'Spending_Score': np.random.uniform(1, 100, n_customers),
        'Gender': np.random.choice(['Male', 'Female'], n_customers),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_customers),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_customers),
        'Num_Children': np.random.poisson(1.5, n_customers),
        'Credit_Score': np.random.normal(650, 100, n_customers).astype(int),
        'Years_Customer': np.random.exponential(3, n_customers).astype(int),
        'Total_Purchases': np.random.poisson(25, n_customers),
        'Avg_Order_Value': np.random.normal(75, 25, n_customers),
        'Days_Since_Last_Purchase': np.random.exponential(30, n_customers).astype(int),
        'Product_Category_Preference': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_customers),
        'Channel_Preference': np.random.choice(['Online', 'Store', 'Both'], n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure realistic constraints
    df['Age'] = np.clip(df['Age'], 18, 80)
    df['Annual_Income'] = np.clip(df['Annual_Income'], 20000, 150000)
    df['Credit_Score'] = np.clip(df['Credit_Score'], 300, 850)
    df['Years_Customer'] = np.clip(df['Years_Customer'], 0, 20)
    df['Days_Since_Last_Purchase'] = np.clip(df['Days_Since_Last_Purchase'], 1, 365)
    
    print("Customer dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df


###############################################################################################################
# Function name :- preprocess_data()
# Description :- Preprocess customer data for clustering
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def preprocess_data(df):
    """Preprocess customer data for clustering"""
    
    # Select numerical features for clustering
    numerical_features = ['Age', 'Annual_Income', 'Spending_Score', 'Num_Children', 
                         'Credit_Score', 'Years_Customer', 'Total_Purchases', 
                         'Avg_Order_Value', 'Days_Since_Last_Purchase']
    
    # Create feature matrix
    X = df[numerical_features].copy()
    
    # Handle categorical features (optional - for advanced analysis)
    categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 
                           'Product_Category_Preference', 'Channel_Preference']
    
    # Encode categorical variables
    le_dict = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
        le_dict[feature] = le
    
    # Add encoded categorical features
    encoded_features = [f'{feature}_encoded' for feature in categorical_features]
    X_encoded = df[encoded_features].copy()
    
    print(f"\nNumerical features: {numerical_features}")
    print(f"Encoded categorical features: {encoded_features}")
    print(f"Total features for clustering: {len(numerical_features + encoded_features)}")
    
    return X, X_encoded, numerical_features, categorical_features, le_dict


###############################################################################################################
# Function name :- find_optimal_clusters()
# Description :- Find optimal number of clusters using multiple methods
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def find_optimal_clusters(X, max_k=10):
    """Find optimal number of clusters using multiple methods"""
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
    
    # Plot optimization results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(k_range, calinski_scores, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Analysis')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "optimal_clusters_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    return optimal_k, inertias, silhouette_scores, calinski_scores


###############################################################################################################
# Function name :- perform_clustering()
# Description :- Perform customer segmentation using multiple algorithms
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def perform_clustering(X, n_clusters):
    """Perform customer segmentation using multiple algorithms"""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # Calculate metrics for both methods
    methods = {
        'K-Means': (kmeans, kmeans_labels),
        'Hierarchical': (hierarchical, hierarchical_labels)
    }
    
    results = {}
    for method_name, (model, labels) in methods.items():
        silhouette_avg = silhouette_score(X_scaled, labels)
        calinski_score = calinski_harabasz_score(X_scaled, labels)
        
        results[method_name] = {
            'model': model,
            'labels': labels,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score
        }
        
        print(f"\n{method_name} Results:")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")
    
    # Choose best method
    best_method = max(results.keys(), key=lambda x: results[x]['silhouette_score'])
    print(f"\nBest clustering method: {best_method}")
    
    return results, best_method, X_scaled, scaler


###############################################################################################################
# Function name :- visualize_segments()
# Description :- Create comprehensive visualizations of customer segments
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def visualize_segments(df, cluster_labels, numerical_features, n_clusters):
    """Create visualizations for customer segments"""
    
    # Add cluster labels to dataframe
    df_segmented = df.copy()
    df_segmented['Segment'] = cluster_labels
    
    # 1. Income vs Spending Score (classic customer segmentation plot)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(df_segmented['Annual_Income'], df_segmented['Spending_Score'], 
                         c=df_segmented['Segment'], cmap='viridis', alpha=0.7)
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.title('Customer Segments: Income vs Spending')
    plt.colorbar(scatter)
    
    # 2. Age vs Spending Score
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(df_segmented['Age'], df_segmented['Spending_Score'], 
                         c=df_segmented['Segment'], cmap='viridis', alpha=0.7)
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.title('Customer Segments: Age vs Spending')
    plt.colorbar(scatter)
    
    # 3. Segment distribution
    plt.subplot(2, 3, 3)
    segment_counts = df_segmented['Segment'].value_counts().sort_index()
    plt.bar(segment_counts.index, segment_counts.values, color='skyblue', edgecolor='black')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.title('Customer Segment Distribution')
    plt.xticks(segment_counts.index)
    
    # 4. Segment characteristics heatmap
    plt.subplot(2, 3, 4)
    segment_means = df_segmented.groupby('Segment')[numerical_features].mean()
    sns.heatmap(segment_means.T, annot=True, cmap='RdYlBu_r', center=0, fmt='.1f')
    plt.title('Segment Characteristics (Mean Values)')
    plt.xlabel('Segment')
    plt.ylabel('Features')
    
    # 5. Gender distribution by segment
    plt.subplot(2, 3, 5)
    gender_segment = pd.crosstab(df_segmented['Segment'], df_segmented['Gender'])
    gender_segment.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Gender Distribution by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.legend(title='Gender')
    plt.xticks(rotation=0)
    
    # 6. Education level distribution by segment
    plt.subplot(2, 3, 6)
    edu_segment = pd.crosstab(df_segmented['Segment'], df_segmented['Education_Level'])
    edu_segment.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Education Level Distribution by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.legend(title='Education Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "customer_segmentation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_segmented


###############################################################################################################
# Function name :- analyze_segments()
# Description :- Analyze customer segment characteristics and profiles
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def analyze_segments(df_segmented, numerical_features, categorical_features):
    """Analyze customer segment characteristics"""
    
    print("\n" + "="*80)
    print("CUSTOMER SEGMENT ANALYSIS")
    print("="*80)
    
    for segment in sorted(df_segmented['Segment'].unique()):
        segment_data = df_segmented[df_segmented['Segment'] == segment]
        
        print(f"\nSEGMENT {segment}:")
        print("-" * 50)
        print(f"Size: {len(segment_data)} customers ({len(segment_data)/len(df_segmented)*100:.1f}%)")
        
        # Numerical characteristics
        print("\nNumerical Characteristics:")
        for feature in numerical_features:
            mean_val = segment_data[feature].mean()
            std_val = segment_data[feature].std()
            print(f"  {feature}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Categorical characteristics
        print("\nCategorical Characteristics:")
        for feature in categorical_features:
            mode_val = segment_data[feature].mode().iloc[0] if not segment_data[feature].mode().empty else 'N/A'
            mode_pct = (segment_data[feature] == mode_val).mean() * 100
            print(f"  {feature}: {mode_val} ({mode_pct:.1f}%)")
        
        # Segment profile summary
        print(f"\nSegment {segment} Profile:")
        avg_income = segment_data['Annual_Income'].mean()
        avg_spending = segment_data['Spending_Score'].mean()
        avg_age = segment_data['Age'].mean()
        
        if avg_income > 60000 and avg_spending > 60:
            profile = "High-Value Customers"
        elif avg_income < 40000 and avg_spending < 40:
            profile = "Budget-Conscious Customers"
        elif avg_income > 60000 and avg_spending < 40:
            profile = "High-Income, Low-Spending Customers"
        elif avg_income < 40000 and avg_spending > 60:
            profile = "Low-Income, High-Spending Customers"
        else:
            profile = "Moderate Customers"
        
        print(f"  Profile: {profile}")
        print(f"  Average Income: ${avg_income:,.0f}")
        print(f"  Average Spending Score: {avg_spending:.1f}")
        print(f"  Average Age: {avg_age:.1f}")


###############################################################################################################
# Function name :- create_business_recommendations()
# Description :- Generate business recommendations based on customer segments
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def create_business_recommendations(df_segmented):
    """Generate business recommendations based on customer segments"""
    
    print("\n" + "="*80)
    print("BUSINESS RECOMMENDATIONS BY SEGMENT")
    print("="*80)
    
    recommendations = {
        0: {
            'name': 'High-Value Customers',
            'strategy': 'Retention & Upselling',
            'actions': [
                'VIP customer service',
                'Premium product recommendations',
                'Loyalty rewards program',
                'Exclusive offers and early access'
            ]
        },
        1: {
            'name': 'Budget-Conscious Customers',
            'strategy': 'Value-Focused Marketing',
            'actions': [
                'Discount offers and promotions',
                'Budget-friendly product bundles',
                'Price comparison tools',
                'Cost-saving tips and guides'
            ]
        },
        2: {
            'name': 'High-Income, Low-Spending',
            'strategy': 'Engagement & Conversion',
            'actions': [
                'Luxury product showcases',
                'Personalized recommendations',
                'Lifestyle-based marketing',
                'Premium service offerings'
            ]
        },
        3: {
            'name': 'Low-Income, High-Spending',
            'strategy': 'Support & Retention',
            'actions': [
                'Flexible payment options',
                'Budget management tools',
                'Financial wellness resources',
                'Loyalty program benefits'
            ]
        }
    }
    
    for segment in sorted(df_segmented['Segment'].unique()):
        if segment in recommendations:
            rec = recommendations[segment]
            print(f"\nSEGMENT {segment} - {rec['name']}:")
            print(f"Strategy: {rec['strategy']}")
            print("Recommended Actions:")
            for i, action in enumerate(rec['actions'], 1):
                print(f"  {i}. {action}")


###############################################################################################################
# Function name :- preserve_model()
# Description :- Save the trained clustering model and scaler
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################

def preserve_model(results, best_method, scaler, le_dict):
    """Save the trained model and scaler"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best model
    model_file = MODELS_DIR / f"customer_segmentation_model_{timestamp}.joblib"
    joblib.dump(results[best_method]['model'], model_file)
    
    # Save scaler
    scaler_file = MODELS_DIR / f"scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_file)
    
    # Save label encoders
    encoders_file = MODELS_DIR / f"label_encoders_{timestamp}.joblib"
    joblib.dump(le_dict, encoders_file)
    
    print(f"\nModel saved at: {model_file}")
    print(f"Scaler saved at: {scaler_file}")
    print(f"Label encoders saved at: {encoders_file}")
    
    return model_file, scaler_file, encoders_file


###############################################################################################################
# Function name :- main()
# Description :- Main function for Customer Segmentation case study
# Author :- Sakshi Ashok Adale
# Date :- 16/03/2026
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation Case Study")
    parser.add_argument("--customers", type=int, default=1000, help="Number of customers to generate")
    parser.add_argument("--max_k", type=int, default=8, help="Maximum k for optimization")
    args = parser.parse_args()
    
    line = "*" * 84
    print(line)
    print("--------------------------- Customer Segmentation Case Study ---------------------------")
    
    # Generate customer data
    df = generate_customer_data(n_customers=args.customers)
    
    # Preprocess data
    X, X_encoded, numerical_features, categorical_features, le_dict = preprocess_data(df)
    
    # Find optimal clusters
    optimal_k, inertias, silhouette_scores, calinski_scores = find_optimal_clusters(X, args.max_k)
    
    # Perform clustering
    results, best_method, X_scaled, scaler = perform_clustering(X, optimal_k)
    
    # Visualize segments
    df_segmented = visualize_segments(df, results[best_method]['labels'], 
                                    numerical_features, optimal_k)
    
    # Analyze segments
    analyze_segments(df_segmented, numerical_features, categorical_features)
    
    # Create business recommendations
    create_business_recommendations(df_segmented)
    
    # Save model
    preserve_model(results, best_method, scaler, le_dict)
    
    print(line)
    print("Customer segmentation completed successfully!")
    print(f"Artifacts saved in: {ARTIFACTS.resolve()}")
    print(line)


if __name__ == "__main__":
    main()