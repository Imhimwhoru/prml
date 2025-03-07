
import numpy as np
import struct

# ======================== LOAD MNIST (UBYTE FORMAT) ========================
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images / 255.0  # Normalize to [0, 1]

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load dataset
x_train = load_mnist_images(r'test/train-images.idx3-ubyte')
y_train = load_mnist_labels(r'test/train-labels.idx1-ubyte')
x_test = load_mnist_images(r'test/t10k-images.idx3-ubyte')
y_test = load_mnist_labels(r'test/t10k-labels.idx1-ubyte')


# ======================== INITIALIZATION ========================
np.random.seed(42)

input_size = 784  # 28x28 pixels
hidden_size1 = 128
hidden_size2 = 64
output_size = 10  # Digits 0-9

# He Initialization
weights = {
    "W1": np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size),
    "b1": np.zeros((1, hidden_size1)),
    "W2": np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1),
    "b2": np.zeros((1, hidden_size2)),
    "W3": np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2),
    "b3": np.zeros((1, output_size))
}

# ======================== ACTIVATION FUNCTIONS ========================
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# ======================== FORWARD PROPAGATION ========================
def forward_propagation(X, weights):
    Z1 = np.dot(X, weights["W1"]) + weights["b1"]
    A1 = relu(Z1)

    Z2 = np.dot(A1, weights["W2"]) + weights["b2"]
    A2 = relu(Z2)

    Z3 = np.dot(A2, weights["W3"]) + weights["b3"]
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

# ======================== LOSS FUNCTION (CROSS ENTROPY + L2) ========================
def compute_loss(Y_pred, Y_true, weights, lambda_=0.01):
    m = Y_true.shape[0]
    log_likelihood = -np.log(Y_pred[range(m), Y_true])
    loss = np.sum(log_likelihood) / m

    # L2 Regularization
    L2_regularization = (lambda_ / (2 * m)) * (
        np.sum(weights["W1"] ** 2) + np.sum(weights["W2"] ** 2) + np.sum(weights["W3"] ** 2)
    )
    return loss + L2_regularization

# ======================== BACKPROPAGATION ========================
def backpropagation(X, Y_true, Z1, A1, Z2, A2, Z3, A3, weights, learning_rate, lambda_):
    m = X.shape[0]
    
    # One-hot encoding of labels
    Y_one_hot = np.zeros((m, output_size))
    Y_one_hot[np.arange(m), Y_true] = 1

    # Compute gradients
    dZ3 = A3 - Y_one_hot
    dW3 = (np.dot(A2.T, dZ3) + lambda_ * weights["W3"]) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, weights["W3"].T)
    dZ2 = dA2 * (A2 > 0)  # ReLU derivative
    dW2 = (np.dot(A1.T, dZ2) + lambda_ * weights["W2"]) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, weights["W2"].T)
    dZ1 = dA1 * (A1 > 0)  # ReLU derivative
    dW1 = (np.dot(X.T, dZ1) + lambda_ * weights["W1"]) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update weights
    weights["W1"] -= learning_rate * dW1
    weights["b1"] -= learning_rate * db1
    weights["W2"] -= learning_rate * dW2
    weights["b2"] -= learning_rate * db2
    weights["W3"] -= learning_rate * dW3
    weights["b3"] -= learning_rate * db3

# ======================== TRAINING LOOP ========================
epochs = 50
batch_size = 64
lambda_ = 0.01  # L2 Regularization factor

for epoch in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[shuffle_indices], y_train[shuffle_indices]

    for i in range(0, x_train.shape[0], batch_size):
        X_batch = x_train[i:i+batch_size]
        Y_batch = y_train[i:i+batch_size]

        # Forward propagation
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X_batch, weights)

        # Backpropagation
        learning_rate = 0.01 / (1 + 0.01 * epoch)  # Learning rate decay
        backpropagation(X_batch, Y_batch, Z1, A1, Z2, A2, Z3, A3, weights, learning_rate, lambda_)

    # Compute loss after each epoch
    _, _, _, _, _, train_pred = forward_propagation(x_train, weights)
    train_loss = compute_loss(train_pred, y_train, weights, lambda_)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# ======================== EVALUATION ========================
def predict(X, weights):
    _, _, _, _, _, A3 = forward_propagation(X, weights)
    return np.argmax(A3, axis=1)

y_pred = predict(x_test, weights)
accuracy = np.mean(y_pred == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
