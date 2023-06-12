# TensorFlow MNIST Classification

This is a code example that demonstrates how to train a simple neural network using TensorFlow and the MNIST dataset for image classification. The model architecture consists of three dense layers, and the code includes data preprocessing steps such as loading the dataset, reshaping the input images, and standardizing the pixel values.

## Prerequisites

To run this code, you'll need the following dependencies:

- TensorFlow
- NumPy
- Matplotlib

## Dataset

The code uses the MNIST dataset, which is a popular benchmark dataset for handwritten digit classification. It contains 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9.

## Code

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display dataset shapes
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Visualize an example image
plt.imshow(x_train[0], cmap='gray')
plt.show()

# Preprocess the labels
y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)

# Reshape the input images
x_train_rs = np.reshape(x_train, (60000, 784))
x_test_rs = np.reshape(x_test, (10000, 784))

# Standardize the pixel values
x_mean = np.mean(x_train_rs)
x_mean2 = np.mean(x_test_rs)
x_std = np.std(x_train_rs)
x_std2 = np.std(x_test_rs)
x_train_std = (x_train_rs - x_mean) / x_std
x_test_std = (x_test_rs - x_mean2) / x_std2

# Define the model architecture
model = Sequential([
    Dense(532, activation='relu', input_shape=(784,)),
    Dense(532, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
model.fit(
    x_train_std,
    y_train_enc,
    epochs=15
)

# Evaluate the model on test set
loss1, accuracy1 = model.evaluate(x_test_std, y_test_enc)

# Evaluate the model on train set
loss2, accuracy2 = model.evaluate(x_train_std, y_train_enc)

# Print accuracies
print('TEST SET ACCURACY:', accuracy1 * 100)
print('TRAIN SET ACCURACY:', accuracy2 * 100)
```

## How to Run

1. Install the required dependencies: TensorFlow, NumPy, and Matplotlib.
2. Save the code to a Python file, e.g., `mnist_classification.py`.
3. Run the Python file: `python mnist_classification.py`.

The code will load the MNIST dataset, preprocess the data, train the model, and evaluate its performance on the test set. Finally, it will print the test set accuracy and train set accuracy.

Feel free to experiment with the code by adjusting the model architecture, hyperparameters, or preprocessing steps to improve the accuracy.
