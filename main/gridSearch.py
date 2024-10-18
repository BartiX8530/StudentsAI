from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.regularizers import l2
import pandas as pd
import numpy as np

# We are using grid search to find the best hyperparameters, such as batch size, number of epochs, optimizer, and number of neurons
# The parameters used and their values are defined in the param_grid dictionary

# Load your dataset
data = pd.read_csv('./data.csv')
X = data.drop(['GradeClass', 'StudentID', 'GPA'], axis=1)  # training data without columns that shouldn't be evaluated
y = data['GPA']  # target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Perform backward feature selection using Linear Regression
lr = LinearRegression()
sfs = SFS(lr, k_features='best', forward=False, floating=False, scoring='neg_mean_squared_error', cv=5)
sfs = sfs.fit(X_train, y_train)
selected_features = list(sfs.k_feature_names_)

# Use only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Define a function to create the model (for KerasRegressor)
def create_model(optimizer='adam', neurons=32, l2_value=0.01):
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    for neuron_count in neurons:
        model.add(Dense(neuron_count, activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(Dense(1, kernel_regularizer=l2(l2_value)))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrap the model using KerasRegressor from scikeras
model = KerasRegressor(model=create_model, verbose=1, l2_value=0.01, neurons=32)

# Define the grid search parameters
param_grid = {
    'batch_size': [12, 16, 32],
    'epochs': [100, 150, 200, 300],
    'optimizer': ['adam', 'rmsprop'],
        'neurons': [
            [32, 32, 32, 32],
            [64, 64, 64, 64],
            [128, 64, 32, 16],
            [128, 128, 64, 32],
            [128, 64, 128, 64, 32],
            [256, 128, 64, 32, 16]
    ]
}

# Create GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Fit the model
grid_result = grid.fit(X_train_scaled, y_train)

# Print the best parameters and score found
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")

# Evaluate the model with the best parameters
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test_scaled)
mse_best_model = mean_squared_error(y_test, y_pred)
print(f'Best Model Mean Squared Error: {mse_best_model}')