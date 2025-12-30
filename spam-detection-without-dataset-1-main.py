import tensorflow as tf # deep learning framework we use to build the model
from tensorflow.keras.preprocessing.text import Tokenizer # converts text into numbers
from tensorflow.keras.preprocessing.sequence import pad_sequences # makes all sequences the same length 
import numpy as np

# =====================================================================
# 1) TRAINING DATA
# =====================================================================
# These are the messages we will use to train the spam detector.
# "1" means the message is SPAM.
# "0" means the message is NOT SPAM.
# In real projects, you will have a large dataset with hundreds/thousands of examples.
texts = [
    "Win a free iPhone",
    "You won cash prize",
    "Let's meet tomorrow",
    "Are you coming to office?",
    "Free entry in contest",
    "Lunch at 1 PM?"
]
labels = [1, 1, 0, 0, 1, 0]

# =====================================================================
# 2) BASIC SETTINGS (HYPERPARAMETERS)
# =====================================================================
# VOCAB_SIZE  = how many unique words the tokenizer should remember.
# MAXLEN      = maximum length of each text after padding/truncation.
# EMBED_DIM   = each word will be converted into a vector of this size.
# EPOCHS      = number of training cycles over the dataset.
VOCAB_SIZE = 500
MAXLEN = 10
EMBED_DIM = 8
EPOCHS = 8

# =====================================================================
# 3) TOKENIZATION: Convert Words → Numbers
# =====================================================================
# Tokenizer learns all words in the training data and assigns each a number.
# Example: "win" -> 4, "free" -> 7, "iphone" -> 9 (numbers vary)
# OOV token = "<OOV>": used for words the model has never seen before.
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")

# Step 1: Learn vocabulary from the training texts
tokenizer.fit_on_texts(texts)

# Step 2: Convert each text message into a list of integer word IDs
# Example: "Win a free iPhone" → [12, 3, 4, 7]  (example numbers)
seqs = tokenizer.texts_to_sequences(texts)

# =====================================================================
# 4) PADDING: Make All Sequences Same Length
# =====================================================================
# Neural networks require inputs of the same length.
# If a sequence is shorter than MAXLEN, we add zeros at the end ("post").
# If it is longer, we cut the extra words from the end.
X = pad_sequences(seqs, maxlen=MAXLEN, padding='post')

# Convert labels into NumPy array for training
y = np.array(labels)

# =====================================================================
# 5) BUILDING THE MODEL
# =====================================================================
# We create a simple neural network:
#   1) Embedding layer: converts word IDs into meaningful vectors
#   2) GlobalAveragePooling1D: averages all word vectors for a message
#   3) Dense layer: learns patterns like "free", "win", "prize" = spam
#   4) Output layer: gives spam probability between 0 and 1
model = tf.keras.Sequential([

    # Step 1: Embedding Layer
    # -----------------------------------------------
    # Converts each number (word ID) into a dense vector of size EMBED_DIM.
    # Example:
    #   Input:  [12, 3, 4, 7, 0, 0, 0, 0, 0, 0]
    #   Output: [[0.1, 0.02, ...], [0.5, -0.3, ...], ...]   (10 vectors)
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,        # number of words to learn
        output_dim=EMBED_DIM,        # size of vector for each word
        input_length=MAXLEN          # each text is padded to this length
    ),

    # Step 2: Global Average Pooling
    # -----------------------------------------------
    # Averages all word vectors into ONE single vector.
    # This reduces the entire sentence into a compact representation.
    # It helps the model focus on "overall meaning" instead of word order.
    tf.keras.layers.GlobalAveragePooling1D(),

    # Step 3: Fully Connected Hidden Layer (Dense)
    # -----------------------------------------------
    # Learns internal patterns from the averaged sentence vector.
    # Activation "relu" keeps positive values and removes negative ones.
    tf.keras.layers.Dense(8, activation='relu'),

    # Step 4: Output Layer
    # -----------------------------------------------
    # Sigmoid outputs a value between 0 and 1.
    # This value = probability that the message is SPAM.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# =====================================================================
# 6) COMPILE THE MODEL
# =====================================================================
# optimizer = 'adam' → adjusts weights automatically
# loss = 'binary_crossentropy' → best for yes/no (spam/not-spam)
# metrics = ['accuracy'] → track how often model is correct
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================================
# 7) TRAINING THE MODEL
# =====================================================================
# The model will learn patterns in the text over 8 epochs.
# verbose=2 → shows training progress in a readable format.
model.fit(X, y, epochs=EPOCHS, verbose=2)

# =====================================================================
# 8) FUNCTION TO PREDICT NEW MESSAGES
# =====================================================================
# This function takes a message (string) and returns spam probability.
def predict_spam(text):

    # Step 1: Convert new text into integer sequence using same tokenizer
    seq = tokenizer.texts_to_sequences([text])

    # Step 2: Pad it so it has the same length as training data
    pad = pad_sequences(seq, maxlen=MAXLEN, padding='post')

    # Step 3: Model predicts probability of being SPAM
    prob = model.predict(pad)[0][0]

    return prob  # return spam probability

# =====================================================================
# 9) TEST THE MODEL WITH NEW EXAMPLES
# =====================================================================
examples = [
    "Congratulations! You have won a free prize",
    "Can we meet tomorrow for lunch?"
]

for ex in examples:
    p = predict_spam(ex)
    label = "SPAM" if p >= 0.5 else "NOT SPAM"
    print(f"'{ex}' -> probability={p:.3f} -> {label}")
