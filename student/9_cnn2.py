# Some important functions and terms used in the code are:

# Sequential: The Sequential API from Keras allows us to define a neural network model as a sequence of layers.

# Conv2D: A convolutional layer applies filters to the input data to extract relevant features.

# MaxPooling2D: A max-pooling layer downsamples the feature maps to reduce spatial dimensions.

# Flatten: A flatten layer converts the multi-dimensional feature maps into a one-dimensional vector.

# Dense: A dense layer is a fully connected layer that performs classification or regression based on the extracted features.

# Compile: The compile method configures the model for training by specifying the optimizer, loss function, and metrics.

# Fit: The fit method trains the model on the training data, optimizing the chosen loss function using backpropagation.

# Evaluate: The evaluate method evaluates the trained model's performance on the test data, calculating the specified metrics.

# Predict: The predict method predicts the output for a given input data.

# argmax: The argmax function returns the indices of the maximum values along a given axis. In this case, it is used to extract the predicted classes and actual labels.
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the Fashion MNIST dataset
# Load the Fashion MNIST dataset: The Fashion MNIST dataset consists of 60,000 grayscale images of fashion items belonging to 10 different classes. It is divided into training and test sets.
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# # Normalize the input images
# Normalize the input images: The pixel values of the images are normalized to a range between 0 and 1 by dividing them by 255. This helps in better training of the model.

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the input data to include the channel dimension
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert the labels to categorical format
# Convert the labels to categorical format: The class labels are represented as integers ranging from 0 to 9. To prepare them for multi-class classification, they are converted to categorical format using one-hot encoding.
# Split the data into training and validation sets: The training data is split into training and validation sets using the train_test_split function from scikit-learn. This allows us to evaluate the model's performance on unseen data.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Define the CNN model
# Define the CNN model: The Convolutional Neural Network (CNN) model is defined using the Sequential API from Keras. It consists of convolutional layers, max-pooling layers, and dense layers. The convolutional layers extract relevant features from the input images, and the dense layers perform classification based on these features.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
# Compile the model: The model is compiled with the Adam optimizer, which is an efficient gradient-based optimization algorithm. The loss function used is categorical cross-entropy, which is suitable for multi-class classification problems. The model is also configured to compute accuracy as a metric.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Train the model: The model is trained on the training data using the fit method. The training is performed in batches, and the number of epochs determines the number of times the model will iterate over the entire training dataset.
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
# Evaluate the model on the test data: After training, the model's performance is evaluated on the test data using the evaluate method. The test loss and accuracy are calculated and printed.
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict classes for the test data
# Predict classes for the test data: The trained model is used to predict the classes for the test data using the predict_classes method. These predicted classes are compared with the actual labels to evaluate the model's predictions.
predictions = np.argmax(model.predict(X_test[:10]), axis=1)
actual_labels = np.argmax(y_test[:10], axis=1)

# Print predictions and actual labels
# Print predictions and actual labels: A few random test samples are chosen, and their predicted classes and actual labels are printed to compare the model's predictions with the ground truth.
for i in range(len(predictions)):
    print(f"Predicted: {predictions[i]}, Actual: {actual_labels[i]}")
