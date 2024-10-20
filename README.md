# Autoencoders

## Introduction

In the realm of unsupervised learning, autoencoders have emerged as powerful tools for data compression, noise removal, and feature extraction. Initially celebrated for their capabilities in image processing, their application has significantly expanded into natural language processing (NLP), addressing the complexities of unstructured text data. This blog post will explore the mechanics of autoencoders, illustrating how they can compress text, clean noisy data, and uncover valuable insights from extensive text corpora. We’ll delve into both the theoretical frameworks and practical implementations, showcasing the transformative potential of autoencoders in the NLP landscape.

## Table of Contents

1. [The Core Architecture of Autoencoders: How Do They Work?](#the-core-architecture-of-autoencoders-how-do-they-work)
2. [Text Compression: Making Language Models Efficient](#text-compression-making-language-models-efficient)
3. [Denoising Autoencoders: Handling Noisy Text Data](#denoising-autoencoders-handling-noisy-text-data)
4. [Feature Extraction with Autoencoders: Unlocking Hidden Knowledge](#feature-extraction-with-autoencoders-unlocking-hidden-knowledge)
5. [Autoencoders and Semi-Supervised Learning: Combining Labeled and Unlabeled Data](#autoencoders-and-semi-supervised-learning-combining-labeled-and-unlabeled-data)
6. [Challenges and Best Practices When Using Autoencoders for Text](#challenges-and-best-practices-when-using-autoencoders-for-text)
7. [Autoencoders for Retrieval-Augmented Generation (RAG)](#autoencoders-for-retrieval-augmented-generation-rag)
8. [Conclusion: The Power of Autoencoders in Text Processing](#conclusion-the-power-of-autoencoders-in-text-processing)


---

## 1. The Core Architecture of Autoencoders: How Do They Work?

Autoencoders are neural networks specifically designed to learn efficient, lower-dimensional representations of input data. This capability allows for effective compression and reconstruction. The key components of an autoencoder include:

- **Encoder**: Transforms the input data into a compressed latent space.
- **Latent Space**: A bottleneck layer that captures the most essential features of the input data.
- **Decoder**: Reconstructs the original data from the latent representation.

In NLP, autoencoders are particularly beneficial for managing high-dimensional text inputs, such as word embeddings or long sequences of text.

### Why Use Autoencoders for Text?

- **Dimensionality Reduction**: Text data is often massive and high-dimensional. Autoencoders distill this data into compact representations, effectively capturing essential patterns while eliminating redundancies.
- **Unsupervised Learning**: Autoencoders learn from raw text data, making them particularly effective when labeled data is scarce or unavailable.

### Types of Autoencoders

- **Basic Autoencoders**: Focus on compression and reconstruction.
- **Denoising Autoencoders (DAEs)**: Specifically trained to remove noise from input data, making them ideal for scenarios involving noisy text.
- **Variational Autoencoders (VAEs)**: Introduce probabilistic elements, enabling applications in text generation and uncertainty representation.

---

## 2. Text Compression: Making Language Models Efficient

Text compression is vital in NLP for handling vast amounts of unstructured data. The encoder compresses text into a low-dimensional latent representation that captures semantic meaning.

### Step-by-Step Process

1. **Tokenization**: Convert raw text into embeddings using methods like Word2Vec or BERT.
2. **Encoding**: The encoder compresses these embeddings into a compact latent space, retaining essential patterns.
3. **Latent Representation**: This compressed vector contains distilled information, often significantly smaller than the original input.
4. **Decoding**: While the focus is on encoding, the decoder reconstructs the original text to evaluate how much of the original meaning is preserved.

### Real-World Application: Text Summarization

Autoencoders can be effectively applied in text summarization, condensing key information from a document into a coherent summary. In our example, we will demonstrate how a simple autoencoder is trained to compress text data, extract key information, and reconstruct meaningful output using **PyTorch Lightning**.

```python
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
```

---

## 3. Denoising Autoencoders: Handling Noisy Text Data

Text data is often noisy, containing errors and irrelevant information. **Denoising Autoencoders (DAEs)** are trained specifically to clean up this noise.

### How DAEs Work

- **Corrupted Input**: During training, noise is artificially added to the input data (e.g., character swaps, random insertions).
- **Reconstruction**: The autoencoder learns to reconstruct clean text from this noisy input.

### Practical Use Case: Preprocessing Data for Sentiment Analysis

Noisy data can hinder the performance of models in tasks like sentiment analysis. Using a DAE to clean this data ensures higher accuracy in subsequent analyses. We can implement this with a practical PyTorch Lightning example where noisy product reviews are processed and cleaned for sentiment analysis.

```python
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
```
---

## 4. Feature Extraction with Autoencoders: Unlocking Hidden Knowledge

Autoencoders can serve as powerful feature extractors. The latent space created by the encoder captures significant patterns that might not be immediately visible in the raw text.

### Using Autoencoders as Feature Extractors

Once trained, you can discard the decoder and utilize the encoder to transform new input data into its compact latent representation. These representations can be leveraged in clustering, classification, or topic modeling tasks.

### Real-World Example: Knowledge Graph Construction

Consider the use of autoencoders to extract key entities and relationships from a corpus of scientific papers. The compact latent representations generated by the encoder can be clustered or mapped into a **knowledge graph**, grouping semantically similar papers or concepts together.

We can demonstrate this by training an autoencoder on a corpus and visualizing the latent features using **t-SNE** or **PCA**, illustrating how autoencoders help organize semantically related texts.

---

## 5. Autoencoders and Semi-Supervised Learning: Combining Labeled and Unlabeled Data

Autoencoders excel in **semi-supervised learning**, where labeled data is limited. By training an autoencoder on extensive unlabeled text corpora, the latent space learned can serve as input features for supervised tasks like text classification.

### Semi-Supervised Sentiment Classification Example

Suppose we only have labeled sentiment data for a small fraction of customer reviews. By first training an autoencoder on the entire corpus (both labeled and unlabeled), we can then use the latent representations of the labeled reviews to train a classifier. This hybrid approach often enhances model performance when labeled data is scarce.

```python
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

```

**OUTPUT**

```
Batch Logits:
 [[-1.875423   1.3197715]
 [ 1.4877982 -1.7081869]]
Predictions: [1 0]
True Labels: [1 0]
```

Understanding the Output

1. **Logits**: 
   - The logits represent the model's raw output scores before any normalization. Each row corresponds to a different input, and each column corresponds to a class (in this case, positive and negative). 
   - For example, in `[[1.4877982, -1.7081869], [-0.6997419, 0.29475552]]`, the first sample has a score of `1.4877982` for the positive class and `-1.7081869` for the negative class.

2. **Predictions**:
   - The predictions are derived from the logits by taking the class with the highest score. 
   - For instance, if the logits for a sample are `[1.4877982, -1.7081869]`, the model will predict `0` (negative) if the highest score corresponds to the negative class. 
   - In the output, `[1 0]` means the model predicted the first sample as positive (class `1`) and the second as negative (class `0`).

3. **True Labels**:
   - The true labels are the actual sentiments for the sentences, which you provided in the dataset. 
   - For example, `[1 0]` means the first sentence is indeed positive, and the second is negative.

- **Correct Predictions**: When the predicted classes match the true labels, the model is making correct predictions. For example, if the model predicts `[1 0]` and the true labels are also `[1 0]`, that means the prediction was correct.
- **Misclassifications**: If there’s a mismatch, like predicting `[1 0]` when the true labels are `[0 1]`, it indicates the model made an error.
  
---

## 6. Challenges and Best Practices When Using Autoencoders for Text

While autoencoders are powerful, they present several challenges:

1. **Text Representation**: Text is inherently discrete, making training autoencoders more difficult. Proper tokenization and embedding strategies (e.g., transformers or custom tokenizers) are essential.
2. **Training Complexity**: The sequential nature of text data makes training autoencoders computationally intensive. Careful tuning of hyperparameters is crucial to avoid overfitting or underfitting.
3. **Overfitting**: Autoencoders can memorize input data instead of learning meaningful representations. Regularization techniques like dropout, early stopping, and adding noise (especially in DAEs) can mitigate this risk.

---

## 7. Autoencoders for Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) combines the ability to retrieve relevant information from an external knowledge base with the generative capabilities of language models. By augmenting generative models with retrieval functionality, RAG systems produce more informed and contextually relevant outputs, especially for complex queries.

### The Role of Autoencoders in RAG

Autoencoders enhance RAG systems by efficiently managing and structuring the knowledge that drives the retrieval component. Let’s break down the encode-decode process within RAG:

1. **Encoding Knowledge into Compressed Latent Representations**: The encoder compresses input data (e.g., documents, research papers) into compact latent space representations. These vectors are smaller than the original documents yet contain essential features, enabling efficient storage and quick retrieval.
2. **Retrieval Based on Latent Space**: When a user query is entered, the encoder compresses it into a latent vector compatible with the pre-encoded documents. This latent query can be matched with the nearest latent representations of the stored data, ensuring retrieval based on semantic similarity rather than simple keyword matching.
3. **Decoding for Knowledge Extraction**: Once relevant latent representations are retrieved, the decoder reconstructs the original text, bringing back specific passages from the compressed latent space. This ensures that the reconstructed data is as faithful as possible to the original, allowing the generative component of the RAG system to combine it with external knowledge for enhanced output.

### The Impact of Autoencoders on RAG Systems

Leveraging autoencoders in RAG systems introduces several key benefits:

1. **Efficient Knowledge Storage and Retrieval**: By compressing large documents into latent vectors, autoencoders facilitate faster lookups during retrieval. This leads to quicker response times in real-time applications like chatbots.
2. **Semantic Matching Over Keyword Matching**: Operating in the latent space enables autoencoders to retrieve documents based on meaning, allowing for context-aware retrieval.
3. **Handling Noisy Queries and Data**: DAEs trained to manage noisy or incomplete queries compress them into cleaner latent representations, allowing the RAG system to focus on essential meaning.
4. **Generalization Across Domains**: Autoencoders capture general text features applicable across multiple fields, enabling RAG systems to generalize information retrieval effectively.

### Detailed Example: Legal Document Retrieval

In a legal assistant system, autoencoders could compress legal cases, statutes, and regulations into latent representations. For example, a lawyer queries: “precedent cases for intellectual property disputes involving software patents.” The process includes:

1. The autoencoder compresses thousands of legal documents into latent vectors.
2. The query is encoded into a latent representation that reflects key legal concepts.
3. The system retrieves relevant documents, even if the exact phrase "software patents" wasn’t used, based on semantic relevance.
4. The decoder reconstructs the most relevant sections for review, providing the lawyer with pertinent information.

### Detailed Example: Product Review Summarization

In e-commerce, an autoencoder-based RAG system compresses product reviews into latent vectors for efficient retrieval and summarization. When a user searches for “best smartphone with long battery life,” the process involves:

1. Encoding reviews into latent representations.
2. Retrieving reviews that mention battery life based on semantic similarity.
3. Decoding relevant reviews and summarizing them for user convenience.

---

## 8. Conclusion: The Power of Autoencoders in Text Processing

Autoencoders offer a flexible and robust framework for compressing text, cleaning noisy data, and extracting valuable features. Whether addressing text summarization, preparing data for sentiment analysis, or constructing knowledge graphs, autoencoders prove to be essential tools in modern NLP. As we continue to explore their capabilities, it’s clear that the power of autoencoders will only grow, enabling us to derive deeper insights from the vast seas of unstructured text data.
