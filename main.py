import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Tensors for the network
x = torch.round(torch.rand(size=[4, 4], dtype=torch.float), decimals=3)
weights = torch.round(torch.rand(size=[4, 4], dtype=torch.float), decimals=3)
bias = torch.round(torch.ones(size=[4, 4], dtype=torch.float), decimals=3)
y_true = torch.round(torch.rand(size=[4, 4], dtype=torch.float), decimals=3)

# Hyperparameters
learning_rate = 0.001
epochs = 3

print(f"Y True:\n{y_true}\n")
print(f"Original Weights:\n{weights}\n\n")

if __name__ == '__main__':

    # Training Loop
    for i in range(epochs):
        # Guess a.k.a. y hat
        y_hat = (weights * x) + bias
        # Mean Squared Error
        mse = torch.round(torch.tensor(data=((y_true - (x * weights)) / 16) ** 2, dtype=torch.float), decimals=3)
        # Updated Weights
        weights += torch.round(torch.tensor(data=(learning_rate * x * (-2 * (y_true - y_hat))), dtype=torch.float), decimals=3)
        bias += torch.round(torch.tensor(data=(learning_rate * (-2 * (y_true - y_hat))), dtype=torch.float), decimals=3)

        print(f"Epoch {i} - Error:\n{mse}\n")
        print(f"Epoch {i} - Updated Weights:\n{weights}")

        '''if ((1.0 - mse) > 0.0) == True:
            print(f"Error is small enough: {mse}\nStopping Training...")
            break

        if ((1.0 - mse) < 0.0):
            print(f"Error is small enough: {mse}\nStopping Training...")
            break'''
