import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax


def subsequent_mask(size):
    """
    Function to compute the mask used in attention layer of decoder

    INPUT:
    size - (int) horizon size

    OUTPUT:
    mask - (torch tensor) boolean array to mask out the data in decoder
    """

    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = torch.from_numpy(mask) == 0

    return mask


def attention(Q, K, V, mask=None, dropout=None):
    """
    Function to compute the attention from given Q, K and V values 

    INPUT:
    Q - (torch tensor) query for the transformer. Shape = (B, H, N, C)
    K - (torch tensor) keys for the transformer. Shape = (B, H, N, C)
    V - (torch tensor) values for the transformer. Shape = (B, H, N, C) 
    mask - (torch tensor) mask for decoder multi head attention layer
    dropout - (float) dropout percentage

    OUTPUT:
    attn_output - (torch tensor) output of the multi head attention layer. Shape = (B, H, N, C)
    """

    # finding the embedding size
    new_emb_size = Q.shape[0]
    # calculating attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(new_emb_size)

    # applying mask on the attention
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)

    # applying softmax layer and calculating prob of attention
    p_attn = softmax(scores, dim=-1)

    # applying dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # multiplying the prob of attentiom with Values (V)  
    attn_output = torch.matmul(p_attn, V)

    return attn_output