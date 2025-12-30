import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# 0. Helper: show current folder and files
# -----------------------------
print("Current working directory:", os.getcwd())
print("Files in this directory:", os.listdir())
print("--------------------------------------------------")

# -----------------------------
# 1. Load & Train CNN on MNIST
# -----------------------------
print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess: reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 digits: 0–9
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training model...")
model.fit(
    x_train, y_train,
    epochs=3,           # you can increase to 5–10 for better accuracy
    batch_size=64,
    validation_data=(x_test, y_test)
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy on MNIST:", test_acc)
print("--------------------------------------------------")

# -----------------------------
# 2. Ask user for image path
# -----------------------------
# Example:
#   - If image is in same folder: digit.png
#   - Full path example: C:/Users/YourName/Desktop/digit.png
img_path = input("Enter path to your digit image (e.g. digit.png): ").strip()

if not os.path.exists(img_path):
    print("❌ File does not exist at path:", img_path)
    print("Make sure the file is in this folder or give full path.")
    raise SystemExit

# -----------------------------
# 3. Load image with OpenCV
# -----------------------------
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("❌ cv2 could not read the image. Check the file type or path.")
    raise SystemExit

print("Original image shape:", img.shape)

# Show original image
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.title("Original image")
plt.axis("off")
plt.show()

# -----------------------------
# 4. Preprocess image like MNIST
# -----------------------------

# Step 1: Resize to 28x28
img_resized = cv2.resize(img, (28, 28))

plt.figure(figsize=(4, 4))
plt.imshow(img_resized, cmap="gray")
plt.title("Resized to 28x28")
plt.axis("off")
plt.show()

# Step 2: Invert colors if background is white (MNIST uses black background)
if np.mean(img_resized) > 127:
    print("Image seems to have white background → inverting colors")
    img_resized = 255 - img_resized

# Step 3: Threshold to clean noise
_, img_thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(4, 4))
plt.imshow(img_thresh, cmap="gray")
plt.title("After threshold / cleaned")
plt.axis("off")
plt.show()

# Step 4: Normalize to [0, 1]
img_norm = img_thresh.astype("float32") / 255.0

# Step 5: Reshape to (1, 28, 28, 1) for CNN
img_input = img_norm.reshape(1, 28, 28, 1)

print("Final input shape to model:", img_input.shape)
print("--------------------------------------------------")

# -----------------------------
# 5. Predict digit with model
# -----------------------------
pred = model.predict(img_input)
predicted_digit = int(tf.argmax(pred, axis=1).numpy()[0])

print("Predicted digit:", predicted_digit)

plt.figure(figsize=(4, 4))
plt.imshow(img_norm.reshape(28, 28), cmap="gray")
plt.title(f"Model sees this\nPredicted: {predicted_digit}")
plt.axis("off")
plt.show()
