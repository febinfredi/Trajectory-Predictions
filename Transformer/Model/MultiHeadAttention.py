import torch
import torch.nn as nn
from Model.utils import attention

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, emb_size, dropout=0.1):

        super(MultiHeadAttention, self).__init__()

        # making sure that the embedding size is divisible by the number
        # of heads
        assert emb_size % num_heads == 0

        # caching values
        self.emb_size = emb_size
        self.num_heads = num_heads

        # creating a single MLP layer for queries, keys and values
        self.q_linear = nn.Linear(emb_size, emb_size)
        self.k_linear = nn.Linear(emb_size, emb_size)
        self.v_linear = nn.Linear(emb_size, emb_size)
        # creating MLP layer for post attention
        self.post_att = nn.Linear(emb_size, emb_size)

        # creating dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # passing the Q, K, and V through 1 layer MLP
        Q, K, V = self.q_linear(Q), self.k_linear(K), self.v_linear(V)  # Shape = (B, N, C)

        # splitting Q, K and V based on num_heads
        batch_size = Q.shape[0]
        new_emb_size = self.emb_size // self.num_heads

        Q = Q.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
        K = K.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
        V = V.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)

        # permuting the dimensions of Q, K and V
        Q = Q.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
        K = K.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
        V = V.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)

        # calculating attention
        attn_output = attention(Q, K, V, mask, self.dropout)            # Shape = (B, H, N, C//H)

        # permuting the dimensions of attn_output and collapsing 
        # the num_heads dimension
        attn_output = attn_output.permute(0,2,1,3)                      # Shape = (B, N, H, C//H)
        attn_output = attn_output.reshape(batch_size, -1, self.emb_size)# Shape = (B, N, C)

        # applying linear layer to output of attention layer
        attn_output = self.post_att(attn_output)                        # Shape = (B, N, C)

        return attn_output