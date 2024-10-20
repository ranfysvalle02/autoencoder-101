import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

# Define a simple autoencoder for text compression
class TextAutoencoder(pl.LightningModule):
    def __init__(self, vocab_size):
        super(TextAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, 2),  # Compress to 2 dimensions
            nn.LeakyReLU(0.001)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, vocab_size),  # Expand back to original size
            nn.Sigmoid()  # Use Sigmoid for output to represent probabilities
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Example dataset: simple text data
text_data = ["abc", "bca", "cab"]

# Create a simple character to index mapping
char_to_index = {char: idx for idx, char in enumerate(sorted(set("".join(text_data))))}
index_to_char = {idx: char for char, idx in char_to_index.items()}
vocab_size = len(char_to_index)

# Convert text to one-hot encoded tensors
def text_to_tensor(text):
    tensor = torch.zeros(len(text), vocab_size)
    for i, char in enumerate(text):
        tensor[i, char_to_index[char]] = 1.0
    return tensor

# Prepare the dataset
data_tensors = torch.stack([text_to_tensor(text) for text in text_data])

# Initialize the autoencoder
autoencoder = TextAutoencoder(vocab_size)
loss_function = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# Training the autoencoder
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    reconstructed_data = autoencoder(data_tensors)
    loss = loss_function(reconstructed_data, data_tensors)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Compress the data
compressed_data = autoencoder.encoder(data_tensors)
print(f"Compressed Data: {compressed_data}")

# Reconstruct the data
reconstructed_data = autoencoder.decoder(compressed_data)
print(f"Reconstructed Data: {reconstructed_data}")

# Convert reconstructed tensors back to characters
def tensor_to_text(tensor):
    indices = torch.argmax(tensor, dim=1)
    return ''.join(index_to_char[idx.item()] for idx in indices)

# Print reconstructed texts
for i, text_tensor in enumerate(reconstructed_data):
    print(f"Original: {text_data[i]}, Reconstructed: {tensor_to_text(text_tensor)}")

"""
Compressed Data: tensor([[[-5.9153e-03, -4.0233e-08],
         [ 4.5033e-02,  3.5911e+00],
         [ 2.9818e+00,  9.8208e-02]],

        [[ 4.5033e-02,  3.5911e+00],
         [ 2.9818e+00,  9.8208e-02],
         [-5.9153e-03, -4.0233e-08]],

        [[ 2.9818e+00,  9.8208e-02],
         [-5.9153e-03, -4.0233e-08],
         [ 4.5033e-02,  3.5911e+00]]], grad_fn=<LeakyReluBackward0>)
Reconstructed Data: tensor([[[9.7110e-01, 1.7838e-02, 3.5556e-02],
         [4.6873e-03, 9.9386e-01, 1.0652e-03],
         [3.9149e-03, 1.7025e-04, 9.9386e-01]],

        [[4.6873e-03, 9.9386e-01, 1.0652e-03],
         [3.9149e-03, 1.7025e-04, 9.9386e-01],
         [9.7110e-01, 1.7838e-02, 3.5556e-02]],

        [[3.9149e-03, 1.7025e-04, 9.9386e-01],
         [9.7110e-01, 1.7838e-02, 3.5556e-02],
         [4.6873e-03, 9.9386e-01, 1.0652e-03]]], grad_fn=<SigmoidBackward0>)
Original: abc, Reconstructed: abc
Original: bca, Reconstructed: bca
Original: cab, Reconstructed: cab
"""
