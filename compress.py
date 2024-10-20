import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

# Step 1: Create a basic Autoencoder class with LeakyReLU
class Autoencoder(pl.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: Compress 3 features to 2
        self.encoder = nn.Sequential(
            nn.Linear(3, 2),
            nn.LeakyReLU(0.001)  # LeakyReLU to avoid dead neurons
        )
        # Decoder: Reconstruct from 2 back to 3
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid()  # Sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 2: Normalize the fruit data to be between 0 and 1
fruit_data = torch.tensor([[7.0, 5.0, 4.0],   
                           [8.0, 4.0, 3.0],    
                           [9.0, 6.0, 1.0]]) / 10.0  # Normalizing by dividing by 10

print(f"Original Data: {fruit_data}")

# Step 3: Create the autoencoder model
autoencoder = Autoencoder()

# Step 4: Set up a loss function and optimizer
loss_function = nn.MSELoss()  # Mean Squared Error for reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)  # Adam optimizer with learning rate

# Step 5: Training the Autoencoder
epochs = 2000  # Number of times we will train the model

for epoch in range(epochs):
    # Zero the gradients (clean slate)
    optimizer.zero_grad()
    
    # Forward pass: compress and then decompress the data
    reconstructed_data = autoencoder(fruit_data)
    
    # Compute the loss (difference between original and reconstructed data)
    loss = loss_function(reconstructed_data, fruit_data)
    
    # Backward pass: calculate gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print loss every 100 epochs for monitoring
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Step 6: After training, let's test the autoencoder

# Compress the data
compressed_data = autoencoder.encoder(fruit_data)
print(f"Compressed Data: {compressed_data}")

# Reconstruct (decompress) the data
reconstructed_data = autoencoder.decoder(compressed_data)
print(f"Reconstructed Data: {reconstructed_data}")
