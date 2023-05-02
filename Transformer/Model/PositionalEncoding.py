import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout=0.1, max_len=5000):

        super(PositionalEncoding, self).__init__()

        # defining the dropout layer
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):


        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = self.dropout(x)

        return x