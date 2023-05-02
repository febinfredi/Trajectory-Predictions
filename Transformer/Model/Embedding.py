import torch
import torch.nn as nn
from Model.PositionalEncoding import PositionalEncoding
import math

class Embeddings(nn.Module):

    def __init__(self, input_size, emb_size):

        super(Embeddings, self).__init__()

        # caching values
        self.emb_size = emb_size

        # creating liner layer for embedding input data
        self.linear_embd = nn.Linear(input_size, emb_size)

        # creating object for positional encoding
        self.pos_encoding = PositionalEncoding(emb_size, dropout=0.1, max_len=5000)
    
    def forward(self, x):

        # creating embeddings for input data
        x = self.linear_embd(x) * math.sqrt(self.emb_size)     # Shape = (B, N, C)
        # incorporating positional embeddings
        x = self.pos_encoding.forward(x)

        return x