# Telco Customer Churn Analysis and Prediction

## Overview

This project analyzes the Telco Customer Churn dataset to predict customer churn (whether a customer will leave the company) using supervised machine learning and to segment customers using unsupervised learning. The goal is to build a robust predictive model for churn and identify customer groups for targeted retention strategies. The project was developed as a practice for a machine learning competition.

## Dataset

The dataset is sourced from Kaggle - Telco Customer Churn. It contains 7,043 records and 21 features, including:

- Demographic Features: gender, SeniorCitizen, Partner, Dependents
- Service Features: PhoneService, InternetService, OnlineSecurity, etc.
- Billing Features: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod
- Target Variable: Churn (Yes/No, indicating if the customer left)

## Data Preprocessing

The dataset was cleaned and prepared for modeling:

- **Handling Missing Values:**  
  TotalCharges had 11 missing values (converted to NaN during type conversion). These were filled with 0, assuming they represent new customers with no charges.

- **Encoding Categorical Variables:**  
  Churn was encoded as binary (Yes=1, No=0).  
  Other categorical columns (e.g., gender, Contract) were one-hot encoded using `pd.get_dummies`, resulting in boolean dummy variables.

- **Removing Irrelevant Features:**  
  `customerID` was dropped as it carries no predictive value.

- **Feature Scaling:**  
  Numerical features (tenure, MonthlyCharges, TotalCharges) were standardized using `StandardScaler` for supervised learning and clustering.

## Machine Learning Models

### 1. Supervised Learning (Churn Prediction)

Two classification models were trained to predict Churn:

- **Logistic Regression:** A simple, interpretable model suitable for binary classification.  
- **Random Forest:** A robust ensemble model effective for imbalanced datasets like churn.

**Methodology:**

- Data was split into 80% training and 20% testing sets.  
- Features were scaled using `StandardScaler`.  
- Performance was evaluated using accuracy and F1 score (better for imbalanced data).

**Results:**

| MODEL               | ACCURACY | F1 SCORE  |
|---------------------|----------|-----------|
| Logistic Regression  | 0.82115  | 0.638968  |
| Random Forest       | 0.82115  | 0.525316  |

**Analysis:**

- Both models achieved identical accuracy (82.12%), but Logistic Regression outperformed Random Forest in F1 score (0.639 vs. 0.525).  
- The higher F1 score for Logistic Regression suggests better balance between precision and recall, likely due to the dataset's moderate imbalance (Churn_Yes ~26%).  
- Random Forest may benefit from hyperparameter tuning (e.g., max_depth, n_estimators).

### 2. Unsupervised Learning (Customer Segmentation)

KMeans Clustering was applied to segment customers into groups based on their features, ignoring Churn.

**Methodology:**

- All features (except `customerID`) were used after one-hot encoding and scaling.  
- KMeans was configured with `n_clusters=3` (arbitrary, can be optimized using the elbow method).  
- Clusters were analyzed to understand customer segments.

**Results:**

- The dataset was divided into three clusters, with approximate sizes: ~3,000, ~2,500, and ~1,543 customers.  
- Cluster characteristics (e.g., average tenure, MonthlyCharges) can be explored using `data_encoded.groupby('Cluster').mean()` to identify distinct customer groups (e.g., loyal vs. high-risk customers).
