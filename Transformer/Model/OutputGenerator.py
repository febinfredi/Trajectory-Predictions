import torch
import torch.nn as nn


class OutputGenerator(nn.Module):

    def __init__(self, emb_size, output_size):

        super(OutputGenerator, self).__init__()

        # creating liner layer for embedding input data
        self.output_gen = nn.Linear(emb_size, output_size)
    
    def forward(self, x):

        x = self.output_gen(x)     # Shape = (B, N, output_size) 

        return x