def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    if abs(error) <= delta:
        return 0.5 * error**2
    else:
        return delta * (abs(error) - 0.5 * delta)

# Feed-forward NN
x = 4
w = 2
b = 1
y_true = 12

y_pred = w*x + b
loss = huber_loss(y_true, y_pred)

print("Prediction:", y_pred)
print("Huber Loss:", loss)

_______________________________________________________________________________________________________________________________________________________________________________________________________________________
# ---------------------------------------------------
# Function to calculate Huber Loss
# ---------------------------------------------------
def huber_loss(y_true, y_pred, delta=1.0):
    # Calculate the error (difference between actual and predicted value)
    error = y_true - y_pred
    
    # If error is small (within delta), use squared loss
    if abs(error) <= delta:
        return 0.5 * error**2
    
    # If error is large, use linear loss (less sensitive to outliers)
    else:
        return abs(error) - 0.5


# ---------------------------------------------------
# Input values (10 samples)
# ---------------------------------------------------
x = [1,2,3,4,5,6,7,8,9,10]          # Input values to the neural network
y_true = [3,6,7,9,12,14,15,18,19,25] # Actual (true) output values


# ---------------------------------------------------
# Feed-forward neural network parameters
# ---------------------------------------------------
w, b = 2, 1   # Weight and bias


# ---------------------------------------------------
# List to store Huber loss for each sample
# ---------------------------------------------------
losses = []


# ---------------------------------------------------
# Forward pass and loss calculation for each sample
# ---------------------------------------------------
for xi, yi in zip(x, y_true):
    # Calculate predicted output using y = wx + b
    y_pred = w * xi + b
    
    # Calculate Huber loss for this sample
    loss = huber_loss(yi, y_pred)
    
    # Store the loss value
    losses.append(loss)


# ---------------------------------------------------
# Display results
# ---------------------------------------------------
print("Huber losses for each sample:", losses)

# Calculate and print mean Huber loss
mean_loss = sum(losses) / len(losses)
print("Mean Huber Loss:", mean_loss)

