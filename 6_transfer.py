##### IMPORTANT: If you run into memory related errors
# Reduce the batch size: Decrease the batch size used during training or evaluation. A smaller batch size will require less memory.
# This code showcases the process of transfer learning using the ImageNet dataset and the VGG16 model. It demonstrates how to leverage a pre-trained model to solve a similar task and provides an evaluation of the model's performance on unseen test data.
# Transfer Learning:
# # Transfer learning is a technique in deep learning where a pre-trained model on a large dataset is used as a starting point for a new task. Instead of training a model from scratch, we leverage the knowledge learned by the pre-trained model and adapt it to a new problem or dataset. This approach is beneficial when the new dataset is small or when we want to solve a similar task.
# ImageNet Dataset:
# The ImageNet dataset is a large-scale image dataset with millions of labeled images from various categories. It has been widely used in computer vision research and benchmarks.

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained VGG16 model
# In this code, the pre-trained model used is VGG16, a popular deep CNN architecture.
# The model is loaded from the Keras library with pre-trained weights from the ImageNet dataset.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Preprocess the data
# The ImageNet dataset requires preprocessing to match the input format expected by the VGG16 model.
# The images are resized to the input size required by VGG16 (typically 224x224 pixels).
# Pixel values are scaled to a range of 0 to 1 and standardized.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = tf.image.resize(X_train, (224, 224)) / 255.0
X_test = tf.image.resize(X_test, (224, 224)) / 255.0

# Create a new model for transfer learning
# The pre-trained VGG16 model is used as the base or backbone model.
# The last few layers of the model are removed, and new layers are added to adapt the model to the new task.
# In this code, a new fully connected layer is added followed by an output layer with the desired number of classes.
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
# Data augmentation is a technique used to artificially increase the size of a training dataset by applying various transformations to the existing data samples. It helps to create more diverse training examples and can improve the model's ability to generalize and handle new data.

# The process of data augmentation involves applying a set of transformations, such as rotation, translation, scaling, flipping, or cropping, to the existing data samples. These transformations introduce variations in the data, making the model more robust and reducing overfitting.

# Data augmentation is commonly used in computer vision tasks, where image data is augmented. TensorFlow provides various tools and functions, such as tf.keras.preprocessing.image.ImageDataGenerator, to perform data augmentation. These tools can generate new samples on-the-fly during training, allowing for a larger and more diverse training set.

# By augmenting the training data, the model can learn from a wider range of variations, leading to improved performance and generalization. However, it's important to note that data augmentation should be used judiciously, considering the characteristics and requirements of the specific problem domain.
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# # Train the model
# The model is compiled with an appropriate optimizer, loss function, and evaluation metric.
# The model is trained on the training dataset, which may require additional pre-processing and data augmentation.
# Training is performed for a specified number of epochs, and the model's performance is monitored.
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# # Evaluate the model
# The trained model is evaluated on the test dataset to measure its performance.
# The loss and accuracy metrics are computed to assess how well the model performs on unseen data.
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions
# The trained model is used to make predictions on a subset of the test dataset.
# The predicted labels are obtained for the input images.
# The predicted labels and the corresponding actual labels are printed for comparison.
predictions = model.predict(X_test[:10])
predicted_labels = tf.argmax(predictions, axis=1)
actual_labels = y_test[:10].flatten()
print("Predicted Labels:", predicted_labels)
print("Actual Labels:", actual_labels)
