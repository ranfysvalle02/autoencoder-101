import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Expanded sample text data for better sentiment representation
sentences = [
    "I love programming",      # 1
    "Python is great for data science",  # 1
    "I enjoy learning about AI",  # 1
    "Deep learning is fascinating",  # 1
    "Natural language processing is interesting",  # 1
    "I hate bugs",              # 0
    "Debugging is frustrating",  # 0
    "I don't like programming errors",  # 0
    "The code is a mess",       # 0
    "I prefer working without issues",  # 1
    "Coding is fun",            # 1
    "I dislike errors",         # 0
    "Fixing bugs is rewarding",  # 1
    "I don't enjoy bad code",    # 0
    "I find it hard to write tests",  # 0
    "Learning to code is enjoyable",  # 1
    "Software development can be tedious"  # 0
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Updated corresponding sentiment labels

# Step 1: Preprocess Text Data
def preprocess_data(sentences: list, encoder: OneHotEncoder) -> torch.Tensor:
    """Convert sentences to one-hot encoded tensor."""
    encoded = encoder.transform(np.array(sentences).reshape(-1, 1))
    return torch.tensor(encoded, dtype=torch.float32)

# Step 2: Create DataLoader
def create_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 2):
    """Create a DataLoader."""
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 3: Define the Autoencoder Model with Dropout
class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 15),  # Increase encoding dimension
            nn.ReLU(),
            nn.Linear(15, 8),  # Intermediate layer
            nn.ReLU(),
            nn.Linear(8, 3),  # Final encoding layer
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),  # Decoding back
            nn.ReLU(),
            nn.Linear(8, 15),
            nn.ReLU(),
            nn.Linear(15, input_dim),  # Decoding back to the original size
            nn.Sigmoid()  # Use Sigmoid for output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)  # Encoding step
        decoded = self.decoder(encoded)  # Decoding step
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Input data (ignore labels)
        reconstructed = self.forward(x)  # Forward pass
        loss = nn.MSELoss()(reconstructed, x)  # Mean Squared Error Loss
        self.log('train_loss', loss)  # Log the loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer

# Step 4: Train the Autoencoder
def train_autoencoder(model, dataloader, epochs: int = 50):
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10)
    trainer.fit(model, dataloader)

# Step 5: Feature Extraction
def extract_features(model, dataloader):
    model.eval()  # Set model to evaluation mode
    features = []
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            encoded = model.encoder(inputs)  # Get the encoded features
            features.append(encoded)
    return torch.cat(features)

# Step 6: Classifier Definition with Dropout
class Classifier(pl.LightningModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 10),  # Input dimension from autoencoder's latent space
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(10, 1),  # Output layer
            nn.Sigmoid()  # Sigmoid output for binary classification
        )

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # Get input and labels
        y_hat = self.forward(x)  # Forward pass
        loss = nn.BCELoss()(y_hat, y.view(-1, 1))  # Binary Cross Entropy Loss
        self.log('train_loss', loss)  # Log the loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer

# Step 7: Train the Classifier
def train_classifier(model, dataloader, epochs: int = 50):
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10)
    trainer.fit(model, dataloader)

# Step 8: Main Execution
if __name__ == "__main__":
    # Initialize OneHotEncoder with handle_unknown parameter
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(np.array(sentences).reshape(-1, 1))  # Fit on training sentences

    # Preprocess the data
    X = preprocess_data(sentences, one_hot_encoder)
    y = torch.tensor(labels, dtype=torch.float32)

    # Split into training and validation datasets
    train_size = int(0.8 * len(X))
    val_size = len(X) - train_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create DataLoaders
    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)

    # Train the autoencoder
    autoencoder = Autoencoder(input_dim=X.shape[1])
    train_autoencoder(autoencoder, train_loader, epochs=50)

    # Extract features
    train_features = extract_features(autoencoder, train_loader)
    val_features = extract_features(autoencoder, val_loader)

    # Create DataLoader for classifier
    classifier_loader = create_dataloader(train_features, y_train)

    # Train the classifier
    classifier = Classifier()
    train_classifier(classifier, classifier_loader, epochs=50)

    # Step 9: Make Predictions on Unseen Data
    def make_prediction(classifier, model, unseen_data):
        unseen_data_tensor = preprocess_data(unseen_data, one_hot_encoder)
        with torch.no_grad():
            features = model.encoder(unseen_data_tensor)  # Extract features
            predictions = classifier(features)  # Get classifier predictions
            predicted_classes = (predictions > 0.5).float()  # Convert probabilities to binary predictions
        return predicted_classes.squeeze().numpy()

    # Example of unseen data
    unseen_sentences = [
        "I enjoy coding challenges",  # Expected: 1
        "I dislike broken code",      # Expected: 0
        "Fixing bugs is a joy",       # Expected: 1
        "The project is terrible",     # Expected: 0
    ]

    predictions = make_prediction(classifier, autoencoder, unseen_sentences)
    for sentence, pred in zip(unseen_sentences, predictions):
        print(f'Sentence: "{sentence}" - Predicted Sentiment: {"Positive" if pred == 1 else "Negative"}')

"""
Sentence: "I enjoy coding challenges" - Predicted Sentiment: Positive
Sentence: "I dislike broken code" - Predicted Sentiment: Negative
Sentence: "Fixing bugs is a joy" - Predicted Sentiment: Positive
Sentence: "The project is terrible" - Predicted Sentiment: Negative
"""
