# ==========================================================
# DIGIT DETECTION USING TENSORFLOW (MNIST DATASET)
# Full Working Code
# ==========================================================

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load the MNIST dataset
# ----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ----------------------------------------------------------
# 2. Preprocess the data
# ----------------------------------------------------------
# Normalize pixel values (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (batch, height, width, channels)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# ----------------------------------------------------------
# 3. Build the CNN model
# ----------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 digit classes
])

# ----------------------------------------------------------
# 4. Compile the model
# ----------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------------------------
# 5. Train the model
# ----------------------------------------------------------
print("Training the model...\n")
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

# ----------------------------------------------------------
# 6. Evaluate on test data
# ----------------------------------------------------------
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# ----------------------------------------------------------
# 7. Predict a single digit from test set
# ----------------------------------------------------------
index = 0  # change to test different samples
img = x_test[index]

plt.imshow(img.reshape(28, 28), cmap="gray")
plt.title("Input Image")
plt.show()

prediction = model.predict(np.array([img]))
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)

# ----------------------------------------------------------
# 8. Save the model
# ----------------------------------------------------------
model.save("digit_detection_model.h5")
print("\nModel saved as digit_detection_model.h5")
