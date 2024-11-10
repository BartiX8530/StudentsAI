import pandas as pd  # Data manipulation and analysis
import numpy as np  # Scientific computing with support for arrays and matrices
from fontTools.varLib.instancer.solver import EPSILON
from sklearn.model_selection import train_test_split  # Split dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # Perform linear regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  # Sequential feature selection
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance
from sklearn.metrics import mean_squared_error  # Compute mean squared error between predicted and actual values
from sklearn.inspection import permutation_importance  # Compute feature importance by permuting feature values
from tensorflow.keras.layers import Input  # Input layer for a neural network
from tensorflow.keras.models import Sequential  # Create a linear stack of layers for a neural network
from tensorflow.keras.layers import Dense  # Create a densely-connected neural network layer
from tensorflow.keras.regularizers import l2  # Apply L2 regularization to neural network weights
from tensorflow.keras.callbacks import EarlyStopping  # Implement early stopping for neural network training
import shap  # SHAP is a game theoretic approach to explain the output of machine learning models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score # Calculate the accuracy of the model

#                   This script is basically used to analyze the SHAP values of the model

# import the model from a .keras file
model = load_model('finishedmodel_037671.keras')

# load the input data
data = pd.read_csv('data.csv')

# leave only columns that are used in the model
selected_features = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music']
X = data[selected_features]
y = data['GPA']  # target column

# Scale the input values using the same scaler
scaler = StandardScaler().fit(X)  # Ensure to use the same scaler used during training
X_scaled = scaler.transform(X)

# Verify selected features
print("Selected features:", selected_features)

# Create a SHAP (Deep)explainer
background = X_scaled[np.random.choice(X_scaled.shape[0], 1000, replace=False)]
explainer = shap.DeepExplainer(model, background)

random_X_test = X_scaled[np.random.choice(X_scaled.shape[0], 1000, replace=False)]

# Calculate the mse of the model
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Calculate the rmse of the model
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Ensure y and y_pred are 1-dimensional and convert y to a NumPy array
y = np.squeeze(y.values)
y_pred = np.squeeze(y_pred)

rmsle = np.sqrt(np.mean(np.square(np.log1p(y_pred + EPSILON) - np.log1p(y + EPSILON))))
print("Root Mean Squared Logarithmic Error:", rmsle)

# Convert RMSLE to multiplicative error
multiplicative_error = np.exp(rmsle) - 1

# Convert to a percentage error
percentage_error = multiplicative_error * 100

print(f"Approximate Relative Percantage Error from RMSE: {rmse / np.mean(y) * 100:.4f}%")

print(f"Approximate Relative Percentage Error from RMSLE: {percentage_error:.4f}%")

smape = np.mean(np.abs(y - y_pred) / ((np.abs(y) + np.abs(y_pred)) / 2)) * 100
print(f"Symmetric Mean Absolute Percentage Error:{smape:.4f}%")

print("Something weird happens next, because some y values are almost 0 and the metrics are incorrectly calculated.")

# Print the minimum and maximum values of y and y_pred for debugging
print("Min value of y:", np.min(y))
print("Max value of y:", np.max(y))
print("Min value of y_pred:", np.min(y_pred))
print("Max value of y_pred:", np.max(y_pred))

# Calculate the RMSPE of the model
rmspe = (np.sqrt(np.mean(np.square((y - y_pred) / (y + EPSILON))))) * 100
print("Root Mean Squared Percentage Error:", rmspe)

mape = np.mean(np.abs((y - y_pred) / (y + EPSILON))) * 100
print("Mean Absolute Percentage Error:", mape)

print("Shape of SHAP values:", np.array(random_X_test).shape)
print("Shape of features:", random_X_test.shape)

shap_values_output = explainer.shap_values(random_X_test)

# Ensure SHAP values are in the correct format
# if isinstance(shap_values, list):
#     correct_shap_values = shap_values_output[0]

reshaped_shap_values = np.concatenate([np.array(vals).reshape(1, -1) for vals in shap_values_output], axis=0)

print("Reshaped SHAP values shape:", reshaped_shap_values.shape)

# Attempt to plot with the reshaped SHAP values
shap.summary_plot(reshaped_shap_values, random_X_test, feature_names=selected_features)

# Force plot for a single prediction (e.g., the first instance in the test set)
# shap.force_plot(explainer.expected_value, reshaped_shap_values, random_X_test[0])

# Ensure SHAP values are correctly shaped and base value is a scalar
base_value = explainer.expected_value.numpy()[0]  # Convert tensor to scalar
reshaped_shap_values = np.squeeze(reshaped_shap_values)  # Shape (1000, 7)

# Select SHAP values and feature values for the first instance (adjust index as needed)
single_shap_values = reshaped_shap_values[0]  # SHAP values for the first instance (shape (7,))
single_data = random_X_test[0]  # Feature values for the first instance (shape (7,))

# Create SHAP waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=single_shap_values,  # SHAP values for a single instance
        base_values=base_value,  # Base value should be a scalar
        data=single_data,  # Corresponding feature values
        feature_names=selected_features  # Ensure correct feature names
    ),
    max_display=len(selected_features),  # To display all features, or set to a fixed number
    show=False  # To display the plot in the notebook
)

plt.tight_layout()
plt.show()

# Ask the user what features they would like a dependence plot for
chosen_feature_names = input(
    f"What features would you like a dependence plot for? (separated by commas, from {selected_features})")
to_choose = [feature for feature in selected_features if feature not in chosen_feature_names]
chosen_interaction_index = input(
    # array of features the user can choose from (not including the chosen features)
    f"What feature would you like to use as the interaction index? 'auto' for (approx.) strongest interaction (from {to_choose})")
chosen_feature_names = [name.strip() for name in chosen_feature_names.split(',')]

# Generate dependence plots for the chosen features
for feature_name in chosen_feature_names:
    if feature_name in selected_features:
        shap.dependence_plot(feature_name, reshaped_shap_values, random_X_test, feature_names=selected_features, interaction_index=chosen_interaction_index)
    else:
        print(f"Feature '{feature_name}' is not in the list of selected features.")