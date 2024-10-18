# this file is used for simplifying the model definitions in other files, and to make them
# easier to use. The function build_model() is used to create a Sequential model with the
# specified parameters, and returns the model, batch size, and number of epochs.

# main/modelBuildUtil.py

import tensorflow as tf
from tensorflow.keras.layers import Input # Input layer for a neural network
from tensorflow.keras.models import Sequential  # Create a linear stack of layers for a neural network
from tensorflow.keras.layers import Dense  # Create a densely-connected neural network layer
from tensorflow.keras.regularizers import l2  # Apply L2 regularization to neural network weights
from tensorflow.keras.models import Sequential # Create a linear stack of layers for a neural network
from tensorflow.keras.layers import Dense # Create a densely-connected neural network layer


def build_model(model_params):
    # Extract the parameters
    batch_size = model_params['batch_size'][0]
    epochs = model_params['epochs'][0]
    optimizer = model_params['optimizer'][0]
    neurons = model_params['neurons'][0]
    input_shape = model_params['input_shape']
    l2_value = model_params['l2 value'][0]

    # Create the Sequential model
    model = Sequential()

    # Add the input layer
    model.add(Input(shape=(input_shape,)))

    # Add layers to the model
    for i, units in enumerate(neurons):
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_value)))

    # Add the output layer
    model.add(Dense(1, activation='linear'))

    # Compile the model, evaluating it using mean squared error
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model, batch_size, epochs