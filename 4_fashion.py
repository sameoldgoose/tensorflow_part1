# Activation Functions: Activation functions introduce non-linearity into neural networks. They determine the output of a neuron given its input. Common activation functions include:

# relu (Rectified Linear Unit): It outputs the input if it's positive, or zero otherwise. It is commonly used in hidden layers.
# softmax: It converts raw scores into probabilities, with each output representing the probability of a specific class. It is commonly used in the output layer for multi-class classification.
# Fashion MNIST Dataset: The Fashion MNIST dataset is a collection of grayscale images (28x28 pixels) representing different fashion items, such as t-shirts, shoes, and dresses. It is a popular benchmark dataset used for image classification tasks.
import tensorflow as tf
from tensorflow import keras

# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
# Preprocessing the data: Before training the model, it is common to preprocess the data to normalize the pixel values. In this case, the pixel values are divided by 255 to scale them between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
# Model Architecture: The model architecture defines the structure and connectivity of the neural network. In this example, we use a sequential model, which is a linear stack of layers. The model consists of three layers:

# Flatten: Reshapes the 2D input images into a 1D array.
# Dense: Fully connected layer with 128 neurons and a ReLU activation function.
# Dense: Fully connected layer with 10 neurons (corresponding to the 10 fashion classes) and a softmax activation function.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# Compilation: Before training the model, it needs to be compiled with specific settings. This involves specifying the optimizer, loss function, and metrics to be used during training.
# Optimizer: The optimizer determines how the model is updated based on the computed gradients during training. The 'adam' optimizer is a popular choice that adapts the learning rate dynamically.
# Loss function: The loss function measures the discrepancy between the predicted output and the true output. In this case, we use 'sparse_categorical_crossentropy', which is suitable for multi-class classification problems
# Metrics: Metrics are used to evaluate the performance of the model during training and testing. In this example, we use 'accuracy' as the metric to measure the classification accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
#  The fit method is used to train the model on the training dataset. The model iteratively adjusts its internal parameters to minimize the loss and improve accuracy.

# Evaluation: After training, the model is evaluated on the test dataset using the evaluate method. The test loss and accuracy are computed and printed.
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Make predictions
# Predictions: The trained model is used to make predictions on new, unseen data (test dataset) using the predict method. The predictions are probabilities indicating the likelihood of each class.
predictions = model.predict(test_images)
