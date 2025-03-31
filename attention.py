import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SimpleSelfAttentionModel(nn.Module):

    def __init__(self, output_size, embeddings, max_length=60, dropout=0.2):
        super().__init__()

        self.n_head = 1
        self.max_length = max_length

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.token_embedding_table = nn.Embedding(num_embeddings, dim)
        self.token_embedding_table = self.token_embedding_table.from_pretrained(
            torch.Tensor(embeddings), freeze=True)
        self.position_embedding_table = nn.Embedding(self.max_length, dim)

        head_size = dim // self.n_head
        self.sa = Head(head_size, dim)
        self.ffwd = FeedFoward(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # TODO: Main-lab-Q3 - define output classification layer
        self.output = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(dim, output_size)
        )

    def forward(self, x):
        B, T = x.shape
        # Infer lengths by counting non-padding tokens (assuming padding token ID is 0)
        lengths = (x != 0).sum(dim=1)
        
        tok_emb = self.token_embedding_table(x)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        # Create mask for valid (non-padding) tokens using inferred lengths
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(2).to(x.dtype)  # (B,T,1)
        
        # Apply mask and compute average
        x = (x * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)  # (B,C)

        logits = self.output(x)  # (B,output_size)
        return logits


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd)
                                    for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadAttentionModel(nn.Module):

    def __init__(self, output_size, embeddings, max_length=60, n_head=3 , dropout=0.2):
        super().__init__()
        
        self.max_length = max_length
        self.n_head = n_head
        
        # Process the embeddings
        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape
        
        # Token embeddings
        self.token_embedding_table = nn.Embedding(num_embeddings, dim)
        self.token_embedding_table = self.token_embedding_table.from_pretrained(
            torch.Tensor(embeddings), freeze=True)
        
        # Position embeddings
        self.position_embedding_table = nn.Embedding(self.max_length, dim)

        # MultiHead attention
        head_size = dim // self.n_head
        self.mba = MultiHeadAttention(n_head, head_size, dim)
        
                # Feed-forward network
        self.ffwd = FeedFoward(dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Output classification layer
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, output_size)

    def forward(self, x):
        B, T = x.shape
        
        # Infer lengths by counting non-padding tokens
        lengths = (x != 0).sum(dim=1)
        
        # Embed tokens and positions
        tok_emb = self.token_embedding_table(x)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        
        # Apply multi-head attention with residual connection
        x = x + self.mha(self.ln1(x))
        
        # Apply feed-forward network with residual connection
        x = x + self.ffwd(self.ln2(x))
        
        # Create mask for valid (non-padding) tokens
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(2).to(x.dtype)  # (B,T,1)
        
        # Apply mask and compute average
        x = (x * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)  # (B,C)
        
        # Apply dropout before classification
        x = self.dropout(x)
        
        # Project to output classes
        logits = self.output(x)
        
        return logits


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_head, head_size, n_embd):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self, output_size, embeddings, max_length=60, n_head=3, n_layer=3, dropout=0.2):
        super().__init__()
        
        self.max_length = max_length
        self.n_head = n_head
        self.n_layer = n_layer
        
        # Process the embeddings
        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape
        
        # Token embeddings
        self.token_embedding_table = nn.Embedding(num_embeddings, dim)
        self.token_embedding_table = self.token_embedding_table.from_pretrained(
            torch.Tensor(embeddings), freeze=True)
        
        # Position embeddings
        self.position_embedding_table = nn.Embedding(self.max_length, dim)
        
        head_size = dim // self.n_head
        self.blocks = nn.Sequential(
            *[Block(n_head, head_size, dim) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim)  # final layer norm

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, output_size)

    def forward(self, x):
        B, T = x.shape
        
        # Infer lengths by counting non-padding tokens
        lengths = (x != 0).sum(dim=1)
        
        # Embed tokens and positions
        tok_emb = self.token_embedding_table(x)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        
        # Apply transformer blocks
        x = self.blocks(x)
        
        # Apply final layer normalization
        x = self.ln_f(x)
        
        # Create mask for valid (non-padding) tokens
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(2).to(x.dtype)  # (B,T,1)
        
        # Apply mask and compute average
        x = (x * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)  # (B,C)
        
        # Apply dropout before classification
        x = self.dropout(x)
        
        # Project to output classes
        logits = self.output(x)
        
        return logits