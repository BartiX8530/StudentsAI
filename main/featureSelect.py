import pandas as pd  # Data manipulation and analysis
import numpy as np  # Scientific computing with support for arrays and matrices
from sklearn.model_selection import train_test_split  # Split dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # Perform linear regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  # Sequential feature selection

# The project uses a linear regression model for feature selection and a neural network model for prediction
# this is to save time and resources as the neural network model is computationally expensive

# Load your dataset and define the training data and target column
data = pd.read_csv('./data.csv')
X = data.drop(['GradeClass', 'StudentID', 'GPA'], axis=1)  # training data without columns that shouldn't be evaluated
y = data['GPA']  # target column

# Split the data into training and testing sets
# test_size is the portion of the data that will be used for testing, here set to 15%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Perform backward feature selection using Linear Regression
lr = LinearRegression()
sfs = SFS(lr, k_features='best', forward=False, floating=False, scoring='neg_mean_squared_error', cv=5)
sfs = sfs.fit(X_train, y_train)
selected_features = list(sfs.k_feature_names_)

# print the selected features
print("Selected features:", selected_features)
