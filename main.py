import torch

# Check to see if a GPU (CUDA compatible) is available for use
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Tensors for the network
x = torch.round(torch.rand(size=[4, 4], dtype=torch.float), decimals=3)
weights = torch.round(torch.rand(size=[4, 4], dtype=torch.float), decimals=3)
bias = torch.round(torch.ones(size=[4, 4], dtype=torch.float), decimals=3)

y_true = torch.round(torch.rand(size=[4, 4], dtype=torch.float), decimals=3)

N = y_true.shape[0] * y_true.shape[1]

# Hyperparameters
learning_rate = 0.001
epochs = 10

# Training function
def train(x, y, weights, bias, learning_rate, epochs, n_total_elements):
    # Training Loop
    for i in range(epochs):
        # Guess a.k.a. y_hat
        y_hat = (weights * x) + bias
        # Mean Squared Error
        mse = torch.round(torch.tensor(data=((y_true - (x * weights)) / n_total_elements) ** 2, dtype=torch.float), decimals=3)
        # Updated Weights
        weights += torch.round(torch.tensor(data=(learning_rate * x * (-2 * (y_true - y_hat))), dtype=torch.float),
                               decimals=3)
        bias += torch.round(torch.tensor(data=(learning_rate * (-2 * (y_true - y_hat))), dtype=torch.float), decimals=3)

        print(f"Epoch {i} - Error:\n{mse}\n")
        print(f"Epoch {i} - Updated Weights:\n{weights}")

print(f"Y True:\n{y_true}\n")
print(f"Original Weights:\n{weights}\n")
print(f"Input Values:\n{x}\n\n")

if __name__ == '__main__':

    train(x, y_true, weights, bias, learning_rate, epochs, N)
