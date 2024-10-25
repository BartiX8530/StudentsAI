import pandas as pd  # Data manipulation and analysis
import numpy as np  # Scientific computing with support for arrays and matrices
from sklearn.model_selection import train_test_split  # Split dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # Perform linear regression
from sklearn.ensemble import RandomForestRegressor  # Random Forest for regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  # Sequential feature selection
import shap  # SHAP is a game theoretic approach to explain the output of machine learning models

# Load your dataset
data = pd.read_csv('data.csv')
X = data.drop(['GradeClass', 'StudentID', 'GPA'], axis=1)  # training data without columns that shouldn't be evaluated
y = data['GPA']  # target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Perform backward feature selection using Random Forest
lr = LinearRegression()
sfs = SFS(lr, k_features='best', forward=False, floating=False, scoring='neg_mean_squared_error', cv=5)
sfs = sfs.fit(X_train, y_train)
selected_features = list(sfs.k_feature_names_)

# Fit the SFS to the training data
sfs = sfs.fit(X_train, y_train)

# Extract the selected features
selected_features = list(sfs.k_feature_names_)

# Print the selected features
print("Selected features:", selected_features)

# Train a new linear regression model using only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
lf_selected = LinearRegression()
lf_selected.fit(X_train_selected, y_train)

# SHAP analysis on the Random Forest model with selected features
explainer = shap.LinearExplainer(lf_selected, X_train_selected)
shap_values = explainer(X_test_selected)

# SHAP summary plot to visualize feature importance
shap.summary_plot(shap_values, X_test_selected)

# Optionally, plot individual feature explanations for a single instance
# shap.plots.waterfall(shap_values[0])  # Explaining the first test sample