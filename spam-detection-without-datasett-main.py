import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

X = np.array([
    [1, 0, 1],  # spam
    [0, 1, 0],  # not spam
    [1, 1, 1],  # spam
    [0, 0, 0],  # not spam
    [0, 0, 1],  # not spam (normal with link)
    [1, 0, 0],  # spam (spam words only)
    [0, 1, 1],  # maybe spam
    [0, 1, 0],  # not spam
    [1, 0, 1],  # spam
    [0, 0, 0],  # not spam
])

# Labels: 1 = SPAM, 0 = NOT SPAM
y = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Build the model
model = models.Sequential([
    layers.Dense(2, activation='relu', input_shape=(3,)),  # hidden layer
    layers.Dense(1, activation='sigmoid')                 # output layer
])

# 4. Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train the model on TRAIN data
model.fit(
    X_train, y_train,
    epochs=100,
    verbose=0  # make 1 if you want to see training output
)

# 6. Evaluate on TEST data
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", acc)

# 7. Try a new email example
# [contains_spam_words, long_mail, contains_link]
x_new = np.array([[1, 0, 1]])  # very spammy: spam words + link

prob = model.predict(x_new)[0][0]
print("Spam probability:", prob)

if prob > 0.5:
    print("Result: SPAM")
else:
    print("Result: NOT SPAM")
