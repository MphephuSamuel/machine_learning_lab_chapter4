# Wine Dataset Feature Analysis & Model Training

## Project Overview
This project demonstrates preprocessing, feature selection, and model training on the UCI Wine dataset. It covers handling missing data, encoding categorical features, scaling, and evaluating models including Logistic Regression, KNN, and Random Forest.

## Features
- **Handling Missing Data:** Drop or impute missing values.  
- **Encoding Categorical Data:** Label encoding for ordinal features; one-hot encoding for nominal features.  
- **Feature Scaling:** Standardization and normalization to improve model performance.  
- **Model Training:** Logistic Regression (with L1 regularization), KNN, Random Forest.  
- **Feature Selection:**  
  - L1 regularization for sparse coefficients.  
  - Sequential Backward Selection (SBS) for important feature subsets.  
  - RandomForest feature importance + SelectFromModel for automated selection.  
- **Evaluation:** Train/test split with stratification, accuracy metrics, coefficient and importance visualization.  
- **Visualization:** Histograms before/after scaling, coefficient vs C plots, SBS accuracy vs number of features, feature importance plots.

## How to Run
1. Install dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.  
2. Load the Wine dataset:  
   ```python
   df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
