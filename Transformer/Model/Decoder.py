import torch
import torch.nn as nn
from Model.MultiHeadAttention import MultiHeadAttention
from copy import deepcopy
class DecoderLayer(nn.Module):

    def __init__(self, emb_size, num_heads, ff_hidden_size, dropout=0.1):

        super(DecoderLayer, self).__init__()

        # creating dropout layer
        self.dropout = nn.Dropout(dropout)

        # creating normalization layer for self attention module
        self.norm_attn = nn.LayerNorm(emb_size)
        # creating normalization layer for encoder-decoder attention module
        self.norm_enc_dec = nn.LayerNorm(emb_size)
        # creating normalization layer for feed forward layer
        self.norm_ff = nn.LayerNorm(emb_size)

        # creating object for multi head self attention layer
        self.attn = MultiHeadAttention(num_heads, emb_size, dropout)
        # creating object for multi head encoder-decoder attention layer
        self.enc_dec_attn = MultiHeadAttention(num_heads, emb_size, dropout)

        # creating feed forward layer
        self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
                                nn.ReLU(), 
                                nn.Dropout(dropout),
                                nn.Linear(ff_hidden_size, emb_size))

    def forward(self, x, enc_output, source_mask, target_mask):

        # sublayer 1: Input -> LayerNorm -> MultiHeadAttention -> Dropout -> ResidualAdd
        x = x + self.dropout(self.attn.forward(self.norm_attn(x),\
            self.norm_attn(x),self.norm_attn(x), target_mask))                          # Shape = (B, N ,C)
        
        # sublayer 2: Input -> LayerNorm -> EncoderDecoderAttention -> Dropout -> ResidualAdd
        x = x + self.dropout(self.enc_dec_attn.forward(self.norm_enc_dec(x),\
            self.norm_enc_dec(enc_output),self.norm_enc_dec(enc_output), source_mask))  # Shape = (B, N ,C)
        
        # sublayer 3: Input -> LayerNorm -> FFN -> Dropout -> ResidualAdd
        x = x + self.dropout(self.ff(self.norm_ff(x)))                                  # Shape = (B, N ,C)

        return x
        
        

class Decoder(nn.Module):
    """
    class for implementing stack of n decoder layers
    """

    def __init__(self, emb_size, num_heads, ff_hidden_size, n, dropout=0.1):

        super(Decoder, self).__init__()

        # creating object for 1 decoder layer
        decoder_obj = DecoderLayer(emb_size, num_heads, ff_hidden_size, dropout)
        # creating stack of n decoder layers
        self.dec_layers = nn.ModuleList([deepcopy(decoder_obj) for _ in range(n)])

        # defining LayerNorm for decoder end
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x, enc_output, source_mask, target_mask):

        for layer in self.dec_layers:
            x = layer.forward(x, enc_output, source_mask, target_mask)      # Shape = (B, N, C)
        
        x = self.norm(x)                                                    # Shape = (B, N, C)

        return x