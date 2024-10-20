import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
"""
An Autoencoder is a type of artificial neural network used for learning effective representations of input data. 
The main idea for a Denoising Autoencoder is to train the network to ignore signal noise.
"""
class SimpleNoisyCharacterDataset(Dataset):
    def __init__(self, chars):
        self.chars = chars
        self.noise_suffix = '*noise*'
        self.char_to_index = self.build_vocab()
        self.index_to_char = {index: char for char, index in self.char_to_index.items()}

    def build_vocab(self):
        vocab = {char: idx + 1 for idx, char in enumerate(self.chars)}
        for char in self.chars:
            noisy_char = self.add_noise(char)
            vocab[noisy_char] = len(vocab) + 1
        return vocab

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, idx):
        clean_char = self.chars[idx]
        noisy_char = self.add_noise(clean_char)
        noisy_index = self.char_to_index[noisy_char]
        clean_index = self.char_to_index[clean_char]
        return noisy_index, clean_index

    def add_noise(self, char):
        return char + self.noise_suffix

class DenoisingAutoencoder(pl.LightningModule):
    def __init__(self, vocab_size):
        super(DenoisingAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.decoder = nn.Linear(32, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.decoder(x)
        return x  

    def training_step(self, batch, batch_idx):
        noisy_indices, clean_indices = batch
        noisy_indices = noisy_indices.unsqueeze(1)
        outputs = self(noisy_indices)
        outputs = outputs.view(-1, outputs.size(-1))
        clean_indices = clean_indices.view(-1)
        loss = nn.CrossEntropyLoss()(outputs, clean_indices)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

def collate_fn(batch):
    noisy_indices, clean_indices = zip(*batch)
    return (
        torch.tensor(noisy_indices, dtype=torch.long),
        torch.tensor(clean_indices, dtype=torch.long)
    )

simple_chars = ["a", "b", "c", "d", "e", "f", "g", "h"]
dataset = SimpleNoisyCharacterDataset(simple_chars)

print("Training Dataset:")
for i in range(len(dataset)):
    noisy_index, clean_index = dataset[i]
    noisy_char = dataset.index_to_char[noisy_index]
    clean_char = dataset.index_to_char[clean_index]
    print(f"Noisy: '{noisy_char}', Clean: '{clean_char}'")

train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = DenoisingAutoencoder(len(dataset.char_to_index) + 1)
trainer = pl.Trainer(max_epochs=100)

print("Starting training...")
trainer.fit(model, train_loader)

print("\nTesting Denoising Autoencoder with Unseen Inputs...")
unseen_chars = ["f", "g", "h"]
for char in unseen_chars:
    noisy_char = dataset.add_noise(char)
    noisy_index = dataset.char_to_index.get(noisy_char)
    if noisy_index is not None:
        reconstructed = model(torch.tensor([[noisy_index]]))
        reconstructed_index = torch.argmax(reconstructed, dim=2).item()
        reconstructed_char = dataset.index_to_char.get(reconstructed_index, "<unknown>")
        print(f"Original: '{char}', Noisy: '{noisy_char}' -> Reconstructed: '{reconstructed_char}'")
    else:
        print(f"Original: '{char}', Noisy: '{noisy_char}' -> Reconstructed: '<unknown>'")

"""
Original: 'f', Noisy: 'f*noise*' -> Reconstructed: 'f'
Original: 'g', Noisy: 'g*noise*' -> Reconstructed: 'g'
Original: 'h', Noisy: 'h*noise*' -> Reconstructed: 'h'
"""
