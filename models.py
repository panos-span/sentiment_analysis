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
        self.hidden_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
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

        # 2 - construct a sentence representation by averaging word embeddings
        # Create a mask to identify non-padding tokens (more efficient than looping)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        
        # Convert mask to same dtype as embeddings for multiplication
        mask = mask.unsqueeze(2).to(embeddings.dtype)

        # Sum the embeddings using the mask to exclude padding tokens
        # Shape: (batch_size, emb_dim)
        sum_embeddings = (embeddings * mask).sum(dim=1)
        # Add small epsilon to avoid division by zero (though it shouldn't happen)
        representations = sum_embeddings / (lengths.float().unsqueeze(1) + 1e-8)
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
        self.hidden_size = 256
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
        Forward pass for the LSTM model (supports bidirectional and truncation).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_length)
            lengths (torch.Tensor): Actual lengths of sequences in the batch (1D tensor)
            
        Returns:
            torch.Tensor: Logits for each class (shape: batch_size, num_classes)
        """
        batch_size, max_length = x.size(0), x.size(1)
        
        # 1. Clamp lengths to ensure they don't exceed max_length
        clamped_lengths = torch.clamp(lengths, max=max_length)
        
        # 2. Embed the input tokens
        embeddings = self.embeddings(x)  # (batch_size, max_length, emb_dim)
        
        # 3. Pack sequences for efficient LSTM processing
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            clamped_lengths.cpu(),  # Must be on CPU for pack_padded_sequence
            batch_first=True,
            enforce_sorted=False
        )
        
        # 4. LSTM forward pass (returns packed_output and hidden states)
        _, (hidden, _) = self.lstm(packed_embeddings)
        
        # ===== OPTION 1: Use last hidden states (original approach) =====
        # Reshape hidden states (num_layers * num_directions, batch, hidden_size)
        num_layers = self.lstm.num_layers
        num_directions = 2 if self.bidirectional else 1
        h_n = hidden.view(num_layers, num_directions, batch_size, self.hidden_size)
        
        # Get the last layer's hidden state
        last_layer_h_n = h_n[-1]  # (num_directions, batch, hidden_size)
        
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_outputs = torch.cat((last_layer_h_n[0], last_layer_h_n[1]), dim=1)
        else:
            last_outputs = last_layer_h_n[0]  # (batch, hidden_size)
        
        representations = last_outputs  # or last_outputs_trunc
        
        # 5. Apply dropout and fully connected layer
        representations = self.dropout(representations)
        logits = self.fc(representations)
        
        return logits