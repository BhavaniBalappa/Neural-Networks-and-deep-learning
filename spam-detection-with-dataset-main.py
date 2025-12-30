import csv
import random
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dataset from spam.csv (label: v1, text: v2)
texts = []
labels = []  # 1 = spam, 0 = ham

with open("spam_100.csv", encoding="latin-1") as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        if len(row) < 2:
            continue
        label_str = row[0]
        message = row[1]

        # Convert label "spam"/"ham" to 1/0
        label = 1 if label_str.strip().lower() == "spam" else 0

        texts.append(message)
        labels.append(label)

print("Total messages:", len(texts))

# 2. Shuffle and train/test split (NO NumPy)
data = list(zip(texts, labels))
random.shuffle(data)

texts_shuffled, labels_shuffled = zip(*data)
texts_shuffled = list(texts_shuffled)
labels_shuffled = list(labels_shuffled)

train_size = int(0.8 * len(texts_shuffled))  # 80% train, 20% test

texts_train = texts_shuffled[:train_size]
labels_train = labels_shuffled[:train_size]

texts_test = texts_shuffled[train_size:]
labels_test = labels_shuffled[train_size:]

print("Train size:", len(texts_train))
print("Test size:", len(texts_test))

# 3. Create tf.data Datasets
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((texts_train, labels_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((texts_test, labels_test)).batch(batch_size)

# 4. Text vectorization layer
max_tokens = 10000
sequence_length = 100

vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)

# Adapt vectorizer on training text only
text_only_train_ds = train_ds.map(lambda x, y: x)
vectorizer.adapt(text_only_train_ds)

# 5. Build the model
model = models.Sequential([
    vectorizer,  # first convert text -> integer sequences

    layers.Embedding(input_dim=max_tokens, output_dim=16, mask_zero=True),
    layers.GlobalAveragePooling1D(),

    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # spam probability
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train the model
history = model.fit(
    train_ds,
    epochs=5,
    validation_data=test_ds
)

# 7. Evaluate on test data
loss, acc = model.evaluate(test_ds)
print("Test accuracy:", acc)

# 8. Try prediction on a new message
new_messages = [
    "Congratulations! You have won a free lottery ticket. Call now!",
    "Hey, are we still meeting for lunch tomorrow?"
]

new_ds = tf.data.Dataset.from_tensor_slices(new_messages).batch(2)
pred_probs = model.predict(new_ds)

for msg, p in zip(new_messages, pred_probs):
    print("\nMessage:", msg)
    print("Spam probability:", float(p[0]))
    if p[0] > 0.5:
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")


Output : Total messages: 100
Train size: 80
Test size: 20
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ text_vectorization                   │ ?                           │     0 (unbuilt) │
│ (TextVectorization)                  │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ embedding (Embedding)                │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling1d             │ ?                           │               0 │
│ (GlobalAveragePooling1D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
Epoch 1/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 3s 233ms/step - accuracy: 0.5125 - loss: 0.6932 - val_accuracy: 0.8500 - val_loss: 0.6877
Epoch 2/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - accuracy: 0.8750 - loss: 0.6859 - val_accuracy: 0.8500 - val_loss: 0.6813
Epoch 3/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step - accuracy: 0.8875 - loss: 0.6792 - val_accuracy: 0.8500 - val_loss: 0.6751
Epoch 4/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step - accuracy: 0.8750 - loss: 0.6726 - val_accuracy: 0.8500 - val_loss: 0.6688
Epoch 5/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.8750 - loss: 0.6655 - val_accuracy: 0.8500 - val_loss: 0.6623
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.8500 - loss: 0.6623
Test accuracy: 0.8500000238418579
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 220ms/step

Message: Congratulations! You have won a free lottery ticket. Call now!
Spam probability: 0.48493996262550354
Prediction: NOT SPAM

Message: Hey, are we still meeting for lunch tomorrow?
Spam probability: 0.4780796468257904
Prediction: NOT SPAM


