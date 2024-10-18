import pandas as pd  # Data manipulation and analysis
import numpy as np  # Scientific computing with support for arrays and matrices
from sklearn.model_selection import train_test_split  # Split dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # Perform linear regression
from sklearn.tree import DecisionTreeRegressor  # Perform regression using decision trees
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  # Sequential feature selection
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance
from sklearn.metrics import mean_squared_error  # Compute mean squared error between predicted and actual values
from sklearn.inspection import permutation_importance  # Compute feature importance by permuting feature values
from tensorflow.keras.layers import Input # Input layer for a neural network
from tensorflow.keras.models import Sequential  # Create a linear stack of layers for a neural network
from tensorflow.keras.layers import Dense  # Create a densely-connected neural network layer
from tensorflow.keras.regularizers import l2  # Apply L2 regularization to neural network weights
from tensorflow.python.keras.callbacks import EarlyStopping
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

# Build the neural network model (Sequential) with L2 regularization to prevent overfitting (penalize large weights)
model_params = {
    'input_shape': X_train_scaled.shape[1],
    'batch_size': [32],
    'epochs': [300],
    'optimizer': ['rmsprop'],
    'l2 value': [0.01],
        'neurons': [
            [32, 32, 32, 32]
        ]
}

# this is the function from our modelBuildUtil.py file that creates the models based on the parameters
model, batch_size, epochs = build_model(model_params)

# Print the model summary
model.summary()

# Implement early stopping if needed
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train, epochs=300, batch_size=32, validation_split=0.15, verbose=1, callbacks=[early_stopping])

# Evaluate the model using permutation importance
result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')

feature_importances = pd.DataFrame({
    'Feature': X_train_selected.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

print(feature_importances)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse_nn = mean_squared_error(y_test, y_pred)
# print(f'Neural Network Mean Squared Error: {mse_nn}')

#                         Baseline models for comparison
# The baseline models help us understand how well our model performs compared to simpler models.
# We will use a Linear Regression model and a Decision Tree model as baselines -
# Linear Regression is the model we used to select features
# Decision Tree is a simple model that can capture non-linear relationships more accurately
# We will evaluate all models using the same selected features and compare their performance

# Baseline Linear Regression model
baseline_model = LinearRegression()
baseline_model.fit(X_train_selected, y_train)

# Evaluating the Linear Regression model
y_pred_baseline = baseline_model.predict(X_test_selected)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)

# Baseline Decision Tree model
dt = DecisionTreeRegressor()
dt.fit(X_train_selected, y_train)

# Evaluating the Decision Tree model
y_pred_dt = dt.predict(X_test_selected)
mse_dt = mean_squared_error(y_test, y_pred_dt)

# Print out the evaluations of the models, features selected by SFS, and feature importances
print(f'Selected features: {selected_features}')
feature_importances = pd.DataFrame({
    'Feature': X_train_selected.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)
print(f'Neural Network Mean Squared Error: {mse_nn}')
print(f'Baseline Linear Regression Mean Squared Error: {mse_baseline}')
print(f'Baseline Decision Tree Mean Squared Error: {mse_dt}')