# This is the main file for the project
from dataclasses import replace
from random import random

# The project therein is a Python AI project built to analyze data on student performance
# And predict grades of other students based on the data.

# The project is divided into 3 main parts:
# 1. main.py: This file contains the main code for the project - the finished product with the best parameters
# 2. gridSearch.py: This file contains the code for hyperparameter tuning using GridSearchCV
# 3. tests.py: This file contains the code for feature selection and permutation importance

# The project uses a linear regression model for feature selection and a neural network model for prediction
# this is to save time and resources as the neural network model is computationally expensive
# while our data sample is small and can be handled by a simpler model, this is mainly suggested as a proof of concept.

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Scientific computing with support for arrays and matrices
from sklearn.model_selection import train_test_split  # Split dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # Perform linear regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  # Sequential feature selection
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance
from sklearn.metrics import mean_squared_error  # Compute mean squared error between predicted and actual values
from sklearn.inspection import permutation_importance  # Compute feature importance by permuting feature values
from tensorflow.keras.layers import Input # Input layer for a neural network
from tensorflow.keras.models import Sequential  # Create a linear stack of layers for a neural network
from tensorflow.keras.layers import Dense  # Create a densely-connected neural network layer
from tensorflow.keras.regularizers import l2  # Apply L2 regularization to neural network weights
from tensorflow.python.keras.callbacks import EarlyStopping # Implement early stopping for neural network training
import shap # SHAP is a game theoretic approach to explain the output of machine learning models
from modelBuildUtil import build_model # This helps us change and modify the model easily

# Load your dataset
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

# Use only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Build the neural network model with L2 regularization to prevent overfitting (penalize large weights)
# model = Sequential()
# Add layers to the model, with the first layer specifying the input dimension
# model.add(Input(shape=(X_train_scaled.shape[1],)))
# model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(1, kernel_regularizer=l2(0.01)))

# Build the neural network model (Sequential) with L2 regularization to prevent overfitting (penalize large weights)
model_params = {
    'input_shape': X_train_scaled.shape[1],
    'batch_size': [16],
    'epochs': [300],
    'optimizer': ['rmsprop'],
    # value of L2 regularization
    'l2 value': [0.01],
    'neurons': [
# These are layers of the model, with the first layer specifying the input dimension
        [128, 64, 32, 16]
    ]
}

# this is the function from our modelBuildUtil.py file that creates the models based on the parameters
model, batch_size, epochs = build_model(model_params)

# Print the model summary
model.summary()

# Implementing early stopping, which stops training when the model stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train, epochs=300, batch_size=16, validation_split=0.15, verbose=1, callbacks=[early_stopping])

# Evaluate the model using permutation importance, which randomly shuffles the values of each feature to determine its importance
result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')

feature_importances = pd.DataFrame({
    'Feature': X_train_selected.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

print(f'feature importances using permutation importance: ')
print(feature_importances)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse_nn = mean_squared_error(y_test, y_pred)
print(f'Neural Network finished model Mean Squared Error: {mse_nn}')

# Extract the first six numbers after the decimal place for our naming convention
mse_str = f"{mse_nn:.6f}".split('.')[1]

# Format the filename
filename = f"finishedmodel_{mse_str}.h5"

### predicting the GPA of a student based on the input values
# our input values are going to be '18' - Study time weekly, '60' - absences, '1' - Tutoring, '1' - ParentalSupport, '0' - Extracurricular, '1' - Sports, '0' - Music
# Which reflect the selected features in the model

# Convert the input values to a DataFrame which the model can understand with the same columns as the training data
input_values = pd.DataFrame([['18', '60', '1', '1', '0', '1', '0']], columns=X_train.columns[:7])

# Ensure the input DataFrame has the same columns as the training data (headers)
input_values = input_values.reindex(columns=X_train.columns, fill_value=0)

# Select the same features as used in training
input_values_selected = input_values[selected_features]

# Scale the input values using the same scaler
input_values_scaled = scaler.transform(input_values_selected)

# Predict the output using the model
prediction = model.predict(input_values_scaled)

# Translate the GPA to a gradeClass
if prediction < 2.0:
    gradeClassPred = 'F'
elif prediction < 2.5:
    gradeClassPred = 'D'
elif prediction < 3.0:
    gradeClassPred = 'C'
elif prediction < 3.5:
    gradeClassPred = 'B'
else:
    gradeClassPred = 'A'

print(f"GPA is {prediction} which is {gradeClassPred} as a grade (gradeClass)")

# Asking the user if they would like to save the model
save_model = input("Would you like to save the model? (Y/N): ")
if save_model.lower() in ['y', 'ye', 'yes']:
    filename = f"finishedmodel_{mse_str}.keras"
    model.save(filename)
    print(f"Model saved as {filename}")

# Asking the user if they would like to analyze the model using SHAP

analyze_model = input("Would you like to analyze the model using SHAP? (Y/N): ")
if analyze_model.lower() in ['y', 'ye', 'yes']:

    # Verify selected features
    print("Selected features:", list(X_train_selected.columns))

    # Create a SHAP (Deep)explainer
    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(model, background)

    # Compute SHAP values for the test data
    shap_values = explainer.shap_values(X_test_scaled)

    correct_shap_values = shap_values[1]

    random_X_test = X_test_scaled[np.random.choice(X_test_scaled.shape[0], 100, replace=False)]

    print("Shape of SHAP values:", np.array(correct_shap_values).shape)
    print("Shape of features:", random_X_test.shape)

    shap_values_output = explainer.shap_values(random_X_test)
    # For our single-output model (the Dense layer has one output neuron), shap_values_output should be a list with one or two elements
    # print(type(shap_values_output))
    # print([np.array(values).shape for values in shap_values_output])

    # Ensure SHAP values are in the correct format
    if isinstance(shap_values, list):
        correct_shap_values = shap_values_output[0]

    reshaped_shap_values = np.concatenate([np.array(vals).reshape(1, -1) for vals in shap_values_output], axis=0)

    print("Reshaped SHAP values shape:", reshaped_shap_values.shape)

    # Attempt to plot with the reshaped SHAP values
    shap.summary_plot(reshaped_shap_values, random_X_test, feature_names=X_train_selected.columns)