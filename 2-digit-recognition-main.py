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
# 1. Load & Train CNN on MNIST (single digit model)
# -----------------------------
print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess: reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Build CNN model (for single digit 0–9)
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
# 2. Ask user for 2-digit image path
# -----------------------------
# Example:
#   - If image is in same folder: two_digits.png
#   - Full path example: C:/Users/YourName/Desktop/two_digits.png
img_path = input("Enter path to your 2-digit image (e.g. two_digits.png): ").strip()

if not os.path.exists(img_path):
    print("❌ File does not exist at path:", img_path)
    print("Make sure the file is in this folder or give full path.")
    raise SystemExit

# -----------------------------
# 3. Load image with OpenCV (grayscale)
# -----------------------------
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("❌ cv2 could not read the image. Check the file type or path.")
    raise SystemExit

print("Original image shape:", img.shape)

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.title("Original 2-digit image")
plt.axis("off")
plt.show()

# -----------------------------
# 4. Split image into left and right digits
# -----------------------------
h, w = img.shape
mid = w // 2   # simple split in the middle

left_img  = img[:, :mid]
right_img = img[:, mid:]

print("Left digit shape:", left_img.shape)
print("Right digit shape:", right_img.shape)

# Show the split images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(left_img, cmap="gray")
plt.title("Left digit (raw)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(right_img, cmap="gray")
plt.title("Right digit (raw)")
plt.axis("off")
plt.show()

# -----------------------------
# 5. Function: preprocess one digit image like MNIST
# -----------------------------
def preprocess_digit(img_digit, title_prefix="Digit"):
    # Step 1: Resize to 28x28
    img_resized = cv2.resize(img_digit, (28, 28))

    # Step 2: Invert colors if background is white (MNIST uses black digit on white bg)
    if np.mean(img_resized) > 127:
        print(f"{title_prefix}: Image seems to have white background → inverting colors")
        img_resized = 255 - img_resized

    # Step 3: Threshold to clean noise
    _, img_thresh = cv2.threshold(
        img_resized, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 4: Normalize to [0, 1]
    img_norm = img_thresh.astype("float32") / 255.0

    # Step 5: Reshape to (1, 28, 28, 1) for CNN
    img_input = img_norm.reshape(1, 28, 28, 1)

    # Show intermediate
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized, cmap="gray")
    plt.title(f"{title_prefix}: Resized 28x28")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_norm.reshape(28, 28), cmap="gray")
    plt.title(f"{title_prefix}: After threshold")
    plt.axis("off")
    plt.show()

    return img_input, img_norm

# Preprocess left and right digit
left_input, left_norm   = preprocess_digit(left_img,  title_prefix="Left digit")
right_input, right_norm = preprocess_digit(right_img, title_prefix="Right digit")

print("Left final input shape:", left_input.shape)
print("Right final input shape:", right_input.shape)
print("--------------------------------------------------")

# -----------------------------
# 6. Predict each digit
# -----------------------------
pred_left  = model.predict(left_input)
pred_right = model.predict(right_input)

left_digit  = int(tf.argmax(pred_left, axis=1).numpy()[0])
right_digit = int(tf.argmax(pred_right, axis=1).numpy()[0])

print("Predicted LEFT digit :", left_digit)
print("Predicted RIGHT digit:", right_digit)

full_number_str = f"{left_digit}{right_digit}"
print("Predicted 2-digit number:", full_number_str)

# Show what model sees
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(left_norm.reshape(28, 28), cmap="gray")
plt.title(f"Left seen as: {left_digit}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(right_norm.reshape(28, 28), cmap="gray")
plt.title(f"Right seen as: {right_digit}")
plt.axis("off")

plt.tight_layout()
plt.show()
