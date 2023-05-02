import torch
import torch.nn as nn
from Model.MultiHeadAttention import MultiHeadAttention
from copy import deepcopy
class EncoderLayer(nn.Module):

    def __init__(self, emb_size, num_heads, ff_hidden_size, dropout=0.1):

        super(EncoderLayer, self).__init__()

        # creating dropout layer
        self.dropout = nn.Dropout(dropout)

        # creating normalization layer for attention module
        self.norm_attn = nn.LayerNorm(emb_size)
        # creating normalization layer for feed forward layer
        self.norm_ff = nn.LayerNorm(emb_size)

        # creating object for multi head attention layer
        self.attn = MultiHeadAttention(num_heads, emb_size, dropout)

        # creating feed forward layer
        self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
                                nn.ReLU(), 
                                nn.Dropout(dropout),
                                nn.Linear(ff_hidden_size, emb_size))
    
    def forward(self, x):

        # sublayer 1: Input -> LayerNorm -> MultiHeadAttention -> Dropout -> ResidualAdd
        x = x + self.dropout(self.attn.forward(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x)))    # Shape = (B, N ,C)

        # sublayer 2: Input -> LayerNorm -> FFN -> Dropout -> ResidualAdd
        x = x + self.dropout(self.ff(self.norm_ff(x)))                                                      # Shape = (B, N ,C)

        return x

class Encoder(nn.Module):

    def __init__(self, emb_size, num_heads, ff_hidden_size, n, dropout=0.1):

        super(Encoder, self).__init__()
        
        # creating object for 1 encoder layer
        encoder_layer_obj = EncoderLayer(emb_size, num_heads, ff_hidden_size, dropout)
        # creating a stack of n encoder layers
        self.enc_layers = nn.ModuleList([deepcopy(encoder_layer_obj) for _ in range(n)])

        # defining LayerNorm for last layer of encoder
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        for layer in self.enc_layers:
            x = layer.forward(x)               # Shape = (B, N, C)
        
        x = self.norm(x)                        # Shape = (B, N, C)

        return x