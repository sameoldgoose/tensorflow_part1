import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Task 1: Load the dataset
# The dataset is a collection of images and their corresponding labels. In this example, we use the MNIST dataset, which consists of grayscale images of hand-drawn digits (0-9).
# The tfds.load() function is used to load the dataset from TensorFlow Datasets (TFDS) library.
# The dataset is split into training and test sets.
dataset_name = 'mnist'  # Example dataset: MNIST
(train_dataset, test_dataset), dataset_info = tfds.load(
    name=dataset_name,
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)

# Task 2: Prepare the dataset
# The dataset needs to be preprocessed before training. In this step, we normalize the pixel values of the images to be in the range [0, 1].
# The normalize_image() function is defined to normalize each image by dividing the pixel values by 255.0, converting them to floats.
num_classes = dataset_info.features['label'].num_classes

# Normalize pixel values to [0, 1]
def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_dataset = train_dataset.map(normalize_image)
test_dataset = test_dataset.map(normalize_image)

# Task 3: Configure the model
# The model represents the neural network architecture that will be trained on the dataset.
# In this example, we use a simple model consisting of a flatten layer, a dense layer with ReLU activation, and an output layer with softmax activation.
# The flatten layer converts the 2D input images into a 1D array.
# The dense layer is a fully connected layer that performs computations on the input data using weights and biases.
# ReLU (Rectified Linear Unit) activation function introduces non-linearity to the model.
# The output layer uses softmax activation to produce a probability distribution over the classes.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# # Task 4: Compile the model
# Before training, the model needs to be compiled with an optimizer, loss function, and metrics.
# The optimizer (e.g., Adam) adjusts the weights of the neural network during training to minimize the loss.
# The loss function (e.g., SparseCategoricalCrossentropy) measures the discrepancy between the predicted and actual labels.
# Metrics (e.g., accuracy) are used to evaluate the performance of the model during training.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Task 5: Train the model
# The model is trained using the training dataset.
# The training dataset is divided into batches (e.g., batch_size = 64) to improve efficiency.
# The fit() function trains the model for a specified number of epochs (e.g., 5), updating the model's weights using backpropagation.
batch_size = 64
train_dataset = train_dataset.batch(batch_size)
model.fit(train_dataset, epochs=5)

# Task 6: Evaluate the model
# After training, we evaluate the model's performance on the test dataset.
# The test dataset is also divided into batches.
# The evaluate() function calculates the loss and accuracy of the model on the test dataset.
test_dataset = test_dataset.batch(batch_size)
loss, accuracy = model.evaluate(test_dataset)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Task 7: Predict on test dataset
# We select a subset of samples from the test dataset (e.g., 10) to make predictions on.
# The take() function is used to select the desired number of samples.
num_samples = 10
test_samples = test_dataset.take(num_samples)

# Task 8: Predict values
# The selected test samples are passed through the trained model to obtain predictions.
# The predict() function returns the predicted probability distributions for each class for the given samples.
predictions = model.predict(test_samples)
predicted_labels = tf.argmax(predictions, axis=1)

# Task 9: Show predicted labels and images
# We visualize the selected test samples along with their predicted and actual labels.
# The imshow() function displays the images.
# The set_title() function sets the title of each subplot to show the predicted and actual labels.
# The axis('off') function removes the axis labels and ticks.
fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 6))
axes = axes.flatten()

for i, (image, label) in enumerate(tfds.as_numpy(test_samples)):
    predicted_label = predicted_labels[len(predicted_labels) - len(test_samples)]
    axes[i].imshow(image[0], cmap='gray')  # Updated: Select the first image in the batch
    axes[i].set_title(f'Predicted: {predicted_label}, Actual: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Additional tasks:
# Task 10: Save the model
# We save the trained model to disk for future use.
# The save() function saves the model in a specified directory.
# We can later load the saved model using the load_model() function.
model.save('my_model')

# Predict using the loaded model:
# We load the saved model and use it to make predictions on the selected test samples again.
# The predict() function is used to obtain predictions from the loaded model.
# Show predicted labels and images from the loaded model:
# We visualize the selected test samples using the loaded model, similar to step 9.
# The imshow() function displays the images.
# The set_title() function sets the title of each subplot to show the predicted and actual labels.
# The axis('off') function removes the axis labels and ticks.
# Task 11: Load the saved model
loaded_model = tf.keras.models.load_model('my_model')

# Task 12: Predict using the loaded model
predictions_loaded = loaded_model.predict(test_samples)

# Task 13: Show predicted labels and images from the loaded model
fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 6))
axes = axes.flatten()

for i, (image, label) in enumerate(tfds.as_numpy(test_samples)):
    predicted_label_loaded = tf.argmax(predictions_loaded[len(predictions_loaded) - len(test_samples)], axis=0)
    axes[i].imshow(image[0], cmap='gray')  # Updated: Select the first image in the batch
    axes[i].set_title(f'Predicted (Loaded): {predicted_label_loaded}, Actual: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
