import tensorflow as tf
import tensorflow_datasets as tfds

# Load the Fashion MNIST dataset
# Use the tfds.load() function to load the Fashion MNIST dataset, consisting of training and test splits.
# Set the split parameter to ['train', 'test'] to load the training and test sets.
# Use the with_info=True argument to also load information about the dataset.
(train_dataset, test_dataset), info = tfds.load('fashion_mnist', split=['train', 'test'], with_info=True, as_supervised=True)

# # Preprocess the dataset
# Define a preprocessing function, preprocess(image, label), to normalize and reshape the images.
# Cast the image data to float32 and normalize it by dividing by 255.0 to scale the pixel values between 0 and 1.
# Reshape the image from a 2D shape (28x28) to a 1D shape (784) to prepare it for input to the RNN model.
# Return the preprocessed image and its label.
# Use the map() function to apply the preprocess() function to both the training and test datasets.
def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, (-1,))
    return image, label

train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Set the hyperparameters

# hidden_units: The number of units in the RNN layer.
# output_units: The number of output classes.
# batch_size: The number of samples per training batch.

hidden_units = 128
output_units = 10
batch_size = 32

# # Build the RNN model
# Create a sequential model using tf.keras.Sequential().
# Add a Reshape layer to reshape the input images from a 1D shape (784) to a 2D shape (28x28).
# Add a SimpleRNN layer with the specified number of hidden_units.
# Add a Dense layer with output_units and softmax activation for classification.
model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(28, 28), input_shape=(784,)),
    tf.keras.layers.SimpleRNN(hidden_units),
    tf.keras.layers.Dense(output_units, activation='softmax')
])

# Compile the model
# Use the compile() function to configure the model for training.
# Set the optimizer parameter to 'adam' for the Adam optimizer.
# Set the loss parameter to 'sparse_categorical_crossentropy' as the loss function for multi-class classification.
# Set the metrics parameter to ['accuracy'] to track the accuracy during training.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# Use the fit() function to train the model on the training dataset.
# Pass the training dataset with batch_size samples per batch.
# Specify the number of training epochs.
# Provide the test dataset as the validation_data for evaluating the model's performance during training.
model.fit(train_dataset.batch(batch_size),
          epochs=5,
          validation_data=test_dataset.batch(batch_size))

# Evaluate the model on the test dataset
# Use the evaluate() function to evaluate the trained model on the test dataset.
# # Compute and store the loss and accuracy values.
# Make predictions and display actual and predicted labels:
# Use the trained model to make predictions on the test dataset.
# Iterate over a few batches of test images and their corresponding labels.
# Use the argmax() function to find the predicted labels with the highest probability.
# Compare and print the actual labels and predicted labels for each example.
loss, accuracy = model.evaluate(test_dataset.batch(batch_size))
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Make predictions on the test dataset
predictions = model.predict(test_dataset.batch(batch_size))

# Show actual labels and predicted labels for a few examples
for images, labels in test_dataset.batch(batch_size).take(5):
    predicted_labels = tf.argmax(model.predict(images), axis=-1)
    for i in range(batch_size):
        print('Actual Label:', labels[i].numpy())
        print('Predicted Label:', predicted_labels[i].numpy())
        print()
