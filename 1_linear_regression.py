import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Boston Housing dataset
# The Boston Housing dataset is loaded using the load_boston function from the sklearn.datasets module. This dataset contains features and target values for housing prices.
# The dataset is split into training and testing sets using train_test_split from sklearn.model_selection. This ensures that we have separate data for training and evaluating our model.
boston = load_boston()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.2, random_state=42)

# Standardize the input features
# The input features (X_train and X_test) are standardized using StandardScaler from sklearn.preprocessing. Standardization scales the features to have zero mean and unit variance, which helps in training the model more effectively.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# We convert the target values y_train and y_test to column vectors using the np.expand_dims function.

# The y_train and y_test arrays initially have shapes (n_samples,), where n_samples is the number of samples in the dataset. By using np.expand_dims with axis=1, we add an extra dimension to the arrays, resulting in shapes (n_samples, 1).

# This transformation is necessary because TensorFlow expects the target values to have a shape of (n_samples, n_outputs) when using them for training or evaluation. In this case, since we are performing linear regression with a single output variable, we set n_outputs to 1.

# By reshaping the target arrays to column vectors, we ensure that the model correctly interprets the target values as individual outputs rather than a single value for each sample.

# Convert the target values to column vectors
y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

# Define the input shape
input_shape = (X_train.shape[1],)

# Defining the Model Architecture and Loss Function:

# The model architecture is defined using tf.keras.Sequential. In this case, we have a single dense layer with 1 unit, which represents the output.
# The loss function is defined as tf.keras.losses.MeanSquaredError(). This calculates the mean squared error between the predicted and target values.

# Define the model architecture using Keras
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Dense(1)
])

# Compile the model
# Compiling a model in TensorFlow involves configuring its training process before it can be trained on the data. When compiling a model, you define the optimizer, loss function, and metrics to be used during training.

# Optimizer: An optimizer determines how the model will be updated based on the computed gradients. It applies an optimization algorithm to minimize the loss function. Examples of optimizers include Adam, RMSprop, and SGD (Stochastic Gradient Descent).

# Loss function: The loss function measures how well the model performs during training. It quantifies the difference between the predicted output and the true output. The choice of loss function depends on the problem type, such as binary classification, multi-class classification, or regression.

# Metrics: Metrics are used to evaluate the performance of the model. They provide additional information during training, but they do not impact the training process itself. Common metrics include accuracy, precision, recall, and F1-score.
# Defining the Optimizer:

# The optimizer is defined using tf.keras.optimizers.SGD (Stochastic Gradient Descent) with a learning rate of 0.01. The optimizer updates the model's weights based on the computed gradients during training.
model.compile(optimizer='sgd', loss='mse')

# Train the model
# Training Loop:
# The training loop runs for a specified number of epochs (10 in this case).
# For each epoch, the loop iterates over the batches in the training dataset.
# A gradient tape is used to record the operations for automatic differentiation, enabling calculation of gradients for the trainable variables.
# The forward pass is performed by passing the batch_x through the model, which produces the predicted values (y_pred).
# The loss value is calculated by comparing the predicted values (y_pred) with the true values (batch_y) using the loss function.
# The gradients of the loss with respect to the model's trainable variables are computed using the tape.
# The optimizer applies the gradients to update the model's weights.
# Validation loss is computed by calculating the mean squared error between the predicted values for the test dataset and the true values.
# The loss values and validation loss are printed for each epoch.
epochs = 1000
model.fit(X_train, y_train, epochs=epochs, verbose=1)

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Make Predictions on the Test Set:
# After the training loop, we use the trained model to make predictions on the test set.
# We run the y_pred operation and provide the test features X_test as input to the feed_dict.
# The predictions are stored in the test_predictions variable.
# Print the Predicted and Actual Values:
# We iterate through the predicted values and corresponding actual values in the test set.
# For each instance, we print the predicted value and the actual value side by side.
# Print the predicted values and corresponding actual values
for i in range(len(test_predictions)):
    print(f"Prediction: {test_predictions[i][0]:.2f}, Actual: {y_test[i][0]}")
