# Recurrent Neural Networks (RNN): RNNs are a type of neural network architecture designed to process sequential data, such as time series or text. They have recurrent connections that allow them to persist information across time steps, making them suitable for tasks that require capturing dependencies and patterns in sequential data.
# IMDB Movie Review Dataset: The IMDB Movie Review dataset is a collection of movie reviews from the Internet Movie Database (IMDB). It is commonly used for sentiment analysis tasks, where the goal is to classify the sentiment (positive or negative) expressed in the reviews.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Set the maximum number of words to be used
max_words = 10000

# Load the IMDB Movie Review dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

# Preprocessing the data:
# num_words: Specifies the maximum number of words to be used in the dataset. Only the most frequently occurring words will be kept, and the rest will be discarded.
# pad_sequences: Pads sequences with zeros or truncates them to have the same length. In this case, the movie reviews are padded or truncated to a length of 500 words.

# Pad sequences to have the same length
max_sequence_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_sequence_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_sequence_length)

# Define the model architecture
# The model architecture consists of three main components:

# Embedding: Converts input sequences into dense vectors of fixed size, allowing the model to learn word representations. Here, each word index is embedded into a 32-dimensional vector.
# SimpleRNN: A simple recurrent layer with 64 units. It processes the input sequences and maintains a hidden state to capture dependencies between words in the text.
# Dense: A fully connected layer with a single unit and a sigmoid activation function. It maps the hidden state to a binary output (positive or negative sentiment).

model = 

# Compile the model
# The model is compiled with specific settings before training:

# optimizer: The 'adam' optimizer is used, which is an efficient optimization algorithm based on adaptive learning rates.
# loss: The 'binary_crossentropy' loss function is used for binary classification tasks, such as sentiment analysis.
# metrics: The 'accuracy' metric is used to measure the model's performance during training and evaluation.
model.compile(optimizer='__',
              loss='____',
              metrics=['___'])

# Train the model
# The fit method is used to train the model on the training dataset. The model learns to minimize the loss function and improve accuracy by adjusting its internal parameters (weights and biases).
batch_size = 128
epochs = 5
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Evaluate the model on the test dataset
# After training, the model is evaluated on the test dataset using the evaluate method. The test loss and accuracy are computed and printed to assess the model's performance.
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Make predictions
# The trained model is used to make predictions on new, unseen movie reviews from the test dataset using the predict method. The predictions are floating-point values between 0 and 1, indicating the predicted sentiment probability (closer to 0 for negative sentiment and closer to 1 for positive sentiment).
predictions = model.predict(X_test)
# Display the actual labels and predicted sentiment for a few examples
for i in range(10):
    # Convert the predicted probability to a sentiment label (0 or 1)
    predicted_label = 1 if predictions[i] >= 0.5 else 0

    # Convert the actual sentiment label to a string (positive or negative)
    actual_label = 'Positive' if y_test[i] == 1 else 'Negative'

    print('Example', i+1)
    print('Predicted:', predicted_label)
    print('Actual:', actual_label)
    print()