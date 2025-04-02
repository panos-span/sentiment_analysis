import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False, hidden_dim=1000):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        embeddings = np.array(embeddings)
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0,  # Use 0 for padding tokens
        ) # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))

        # 3 - define if the embedding layer will be frozen or finetuned
        # (trainable) during training
        self.embedding_layer.weight.requires_grad = trainable_emb # EX4

        # 4 - define a non-linear transformation of the representations
        embedding_dim = embeddings.shape[1]
        self.hidden_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU()
        )

        # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.output_layer = nn.Linear(hidden_dim, output_size) # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function defines how the data passes through the network.

        Args:
            x (torch.Tensor): Input tensor of token IDs with shape (batch_size, max_length)
            lengths (torch.Tensor): Actual lengths of each sequence in the batch

        Returns: 
            torch.Tensor: The logits for each class
        """

        # 1 - embed the words using the embedding layer
        # Shape: (batch_size, max_length, emb_dim)
        embeddings = self.embedding_layer(x)
        
        # 2 - construct a sentence representation by combining mean and max pooling
        # Create a mask to identify non-padding tokens
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        
        # Convert mask to same dtype as embeddings for multiplication
        mask = mask.unsqueeze(2).to(embeddings.dtype)
        
        # Sum the embeddings using the mask to exclude padding tokens
        # Shape: (batch_size, emb_dim)
        sum_embeddings = (embeddings * mask).sum(dim=1)
        
        # Divide by the actual lengths to get the mean
        mean_embeddings = sum_embeddings / (lengths.float().unsqueeze(1) + 1e-8)
        
        # Max pooling - apply mask by setting padded positions to a very negative value
        padded_embeddings = embeddings.clone()
        # Invert the mask to get padded positions and multiply by a large negative number
        padded_embeddings = padded_embeddings.masked_fill_(~mask.bool(), float('-inf'))
        # Get the maximum value for each feature across the sequence dimension
        max_embeddings = torch.max(padded_embeddings, dim=1)[0]
        
        # Concatenate mean and max pooling results along the feature dimension
        # Shape: (batch_size, 2*emb_dim)
        representations = torch.cat([mean_embeddings, max_embeddings], dim=1)
        
        # 3 - apply non-linear transformation to get new representations
        representations = self.hidden_layer(representations)
        
        # 4 - project the representations to class logits
        logits = self.output_layer(representations)
        
        return logits


class LSTM(nn.Module):
    def __init__(
        self, output_size, embeddings, trainable_emb=False, bidirectional=False, dropout=0.0
    ):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 3
        self.bidirectional = bidirectional

        self.representation_size = (
            2 * self.hidden_size if self.bidirectional else self.hidden_size
        )

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(
            dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,  # Important: set batch_first=True
            dropout=dropout if self.num_layers > 1 else 0,
        )

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True
            )

        # Add dropout before the output layer
        self.dropout = nn.Dropout(dropout)
        
        # Create the output layer
        self.fc = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        """
        Forward pass for the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of token IDs with shape (batch_size, max_length)
            lengths (torch.Tensor): Actual lengths of each sequence in the batch
            
        Returns:
            torch.Tensor: The logits for each class
        """
        # Embed the words using the embedding layer
        # Shape: (batch_size, max_length, emb_dim)
        embeddings = self.embeddings(x)
        
        # Pack the padded sequence for the LSTM
        # This step is crucial for handling variable-length sequences efficiently
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, 
            lengths.cpu(), 
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass with packed input
        _, (hidden, _) = self.lstm(packed_embeddings)

        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # Reshape h_n to (num_layers, num_directions, batch, hidden_size)
        num_layers = self.lstm.num_layers
        num_directions = 2 if self.bidirectional else 1
        h_n = hidden.view(num_layers, num_directions, x.size(0), self.hidden_size)

        # Get the last layer's hidden state
        last_layer_h_n = h_n[-1]  # Shape: (num_directions, batch, hidden_size)

        if self.bidirectional:
            # Concatenate the hidden states from both directions
            last_outputs = torch.cat((last_layer_h_n[0], last_layer_h_n[1]), dim=1)
        else:
            last_outputs = last_layer_h_n[0]

        # Apply dropout
        last_outputs = self.dropout(last_outputs)

        # Pass through fully connected layer
        return self.fc(last_outputs)