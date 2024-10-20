import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Step 1: Create a simple dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts  # List of sentences (strings)
        self.labels = labels  # List of corresponding labels (1 for positive, 0 for negative)

    def __len__(self):
        return len(self.texts)  # Return the number of samples

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]  # Return the sentence and its label

# Sample data (simple sentences and their sentiment labels)
texts = [
    "I love this product!",  # Positive sentiment
    "This is the worst thing ever.",  # Negative sentiment
    "I am very happy with my purchase.",  # Positive sentiment
    "I hate this.",  # Negative sentiment
    "Fantastic service!",  # Positive sentiment
    "Not what I expected."  # Negative sentiment
] * 10  # Replicating to increase size

# Corresponding binary labels (1 for positive, 0 for negative)
labels = [1, 0, 1, 0, 1, 0] * 10  # Replicating to increase size

# Step 2: Create a more complex model for text classification
class SimpleTextClassifier(pl.LightningModule):
    def __init__(self):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(50, 16)  # Simple embedding layer
        self.fc1 = nn.Linear(16, 32)  # First fully connected layer
        self.fc2 = nn.Linear(32, 2)  # Second fully connected layer for output

    def forward(self, x):
        x = self.embedding(x)  # Convert the words into numbers
        x = x.mean(dim=1)  # Average the embeddings for simplicity
        x = torch.relu(self.fc1(x))  # Apply first layer with ReLU activation
        x = self.fc2(x)  # Get our prediction from the second layer
        return x

    def training_step(self, batch, batch_idx):
        texts, labels = batch  # Get the sentences and their labels
        logits = self(texts)  # Get the model's predictions
        loss = nn.CrossEntropyLoss()(logits, labels)  # Calculate how wrong the predictions were
        self.log('train_loss', loss)  # Keep track of the loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)  # Set up how to change the model based on loss

# Step 3: Prepare the data and create a DataLoader
# All sequences must be the same length
text_indices = torch.tensor([
    [1, 2, 3, 0],  # "I love this" (0 is padding)
    [4, 5, 6, 0],  # "This is the worst" (0 is padding)
    [1, 2, 7, 0],  # "I am very happy" (0 is padding)
    [8, 9, 0, 0],  # "I hate this." (0s for padding)
    [10, 11, 0, 0],  # "Fantastic service!" (0s for padding)
    [12, 13, 0, 0]   # "Not what I expected." (0s for padding)
] * 10)  # Replicating to increase size
labels_tensor = torch.tensor(labels * 10)  # Replicating to increase size

# Create dataset and DataLoader
dataset = TextDataset(text_indices, labels_tensor)  # Create our dataset
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)  # Create a loader to get batches of data

# Step 4: Train the model
model = SimpleTextClassifier()  # Create an instance of our model
trainer = pl.Trainer(max_epochs=10)  # Set up how many times to go through the data

# Start training
trainer.fit(model, train_loader)

# Step 5: Test the classifier and print logits and predictions
with torch.no_grad():
    for batch in train_loader:  # Go through each batch in our DataLoader
        texts, true_labels = batch  # Get the sentences and their true labels
        logits = model(texts)  # Get the model's predictions
        predictions = torch.argmax(logits, dim=1)  # Find out which prediction is higher (0 or 1)

        # Print logits, predictions, and true labels for clarity
        print("\nBatch Logits:\n", logits.numpy())  # Logits from the classifier
        print(f'Predictions: {predictions.numpy()}')  # Predicted classes (0 or 1)
        print(f'True Labels: {true_labels.numpy()}')  # True labels (0 or 1)
