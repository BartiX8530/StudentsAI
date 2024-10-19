import pandas as pd  # Data manipulation and analysis
import numpy as np  # Scientific computing with support for arrays and matrices
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

#                   This script is used to analyze the SHAP values of the model
#

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
background = X_scaled[np.random.choice(X_scaled.shape[0], 100, replace=False)]
explainer = shap.DeepExplainer(model, background)

# Compute SHAP values for the test data
shap_values = explainer.shap_values(X_scaled)

correct_shap_values = shap_values[1]

random_X_test = X_scaled[np.random.choice(X_scaled.shape[0], 100, replace=False)]

print("Shape of SHAP values:", np.array(correct_shap_values).shape)
print("Shape of features:", random_X_test.shape)

shap_values_output = explainer.shap_values(random_X_test)

# Ensure SHAP values are in the correct format
if isinstance(shap_values, list):
    correct_shap_values = shap_values_output[0]

reshaped_shap_values = np.concatenate([np.array(vals).reshape(1, -1) for vals in shap_values_output], axis=0)

print("Reshaped SHAP values shape:", reshaped_shap_values.shape)

# Attempt to plot with the reshaped SHAP values
shap.summary_plot(reshaped_shap_values, random_X_test, feature_names=selected_features)

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