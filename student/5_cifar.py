# The code demonstrates the process of building and training a CNN model on the CIFAR-10 dataset, as well as evaluating the model's performance and making predictions on unseen test data.
#ALGORITHM:
# The code builds and trains a Convolutional Neural Network (CNN) model using the CIFAR-10 dataset.
# The CIFAR-10 dataset consists of 60,000 32x32 color images, categorized into 10 different classes, such as airplanes, automobiles, birds, cats, etc.
# The code loads the dataset, normalizes the pixel values, and converts the labels into one-hot encoded vectors.
# It defines a CNN model architecture using various layers such as convolutional layers, max-pooling layers, and dense layers.
# The model is compiled with appropriate optimizer, loss function, and evaluation metric.
# The model is trained on the training data for a specified number of epochs.
# After training, the model is evaluated on the test dataset to measure its performance in terms of loss and accuracy.
# The code then uses the trained model to make predictions on a subset of the test dataset.
# It prints the predicted labels and the corresponding actual labels for comparison.
# The printed labels provide an indication of how well the model is performing in classifying the images from the CIFAR-10 dataset.

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load the CIFAR-10 dataset: The CIFAR-10 dataset consists of 60,000 32x32 color images of 10 different classes, with 6,000 images per class. We use the cifar10.load_data() function from the tensorflow.keras.datasets module to load the dataset. It returns two tuples, one for training data and labels, and the other for test data and labels.

# Normalize pixel values: Similar to the previous examples, we normalize the pixel values to the range of 0 to 1 by dividing each value by 255.
X_train = 
X_test = 

# Convert labels to one-hot encoding: We convert the labels into one-hot encoded vectors using tf.keras.utils.to_categorical, as done in the previous examples.
y_train = 
y_test = 

# Define the CNN model architecture: The CNN model consists of several layers. We start with a 2D convolutional layer (Conv2D) with 32 filters, each of size 3x3, and ReLU activation. This is followed by a max-pooling layer (MaxPooling2D) with a pool size of 2x2 to reduce the spatial dimensions. We repeat this pattern with another convolutional layer and max-pooling layer. Then, we add one more convolutional layer and flatten the output. We continue with two fully connected (Dense) layers with ReLU activation. The final dense layer has 10 units with softmax activation, representing the 10 classes in CIFAR-10.
model = 

# Compile the model: Similar to before, we compile the model by specifying the optimizer, loss function, and evaluation metric. In this case, we use the Adam optimizer, categorical cross-entropy loss, and track accuracy as the metric.
model.compile(optimizer='___',
              loss='______',
              metrics=['accuracy'])

# Train the model: We use the fit method to train the model on the training data, specifying the number of epochs and batch size.
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model: After training, we evaluate the model's performance on the test dataset using the evaluate method, which computes the loss and accuracy of the model on the unseen test data.
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Predict values for the test dataset: We use the predict method to make predictions on the test dataset. We retrieve the predicted labels by taking the argmax along the predicted probabilities axis.
predictions = model.predict(X_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

# Print predicted and actual labels: Finally, we print the predicted labels and actual labels for the first 10 examples in the test dataset. The predicted labels are mapped to their corresponding class names using a predefined list of class names.
actual_labels = np.argmax(y_test[:10], axis=1)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("Predicted Labels:")
for i in range(10):
    print(f"Example {i+1}: {class_names[predicted_labels[i]]}")
print("\nActual Labels:")
for i in range(10):
    print(f"Example {i+1}: {class_names[actual_labels[i]]}")
