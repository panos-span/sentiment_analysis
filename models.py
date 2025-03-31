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

    def __init__(self, output_size, embeddings, trainable_emb=False):
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
        hidden_dim = 128  # We can choose any dimension here
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
        
        # Divide by the actual lengths to get the mean
        # Add small epsilon to avoid division by zero (though it shouldn't happen)
        representations = sum_embeddings / (lengths.float().unsqueeze(1) + 1e-8)
        
        # 3 - apply non-linear transformation to get new representations
        representations = self.hidden_layer(representations)
        
        # 4 - project the representations to class logits
        logits = self.output_layer(representations)
        
        return logits


class LSTM(nn.Module):
    def __init__(
        self, output_size, embeddings, trainable_emb=False, bidirectional=False
    ):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
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
        )

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True
            )

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        representations = ...

        logits = self.linear(representations)

        return logits
