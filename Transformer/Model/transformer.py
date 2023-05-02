
import torch
import torch.nn as nn
from Model.Embedding import Embeddings
from Model.Encoder import Encoder
from Model.Decoder import Decoder
from Model.OutputGenerator import OutputGenerator

class Transformer(nn.Module):


    def __init__(self, encoder_ip_size, decoder_ip_size, model_op_size, emb_size, \
                num_heads, ff_hidden_size, n, dropout=0.1):
 
        super(Transformer, self).__init__()

        # creating embeddings for encoder input
        self.encoder_embedding = Embeddings(encoder_ip_size, emb_size)
        # creating embeddings for decoder input
        self.decoder_embeddings= Embeddings(decoder_ip_size, emb_size)
        
        # creating encoder block
        self.encoder_block = Encoder(emb_size, num_heads, ff_hidden_size, n, dropout)
        # creating decoder block
        self.decoder_block = Decoder(emb_size, num_heads, ff_hidden_size, n, dropout)

        # creating output generator
        self.output_gen = OutputGenerator(emb_size, model_op_size)
    
    def forward(self, enc_input, dec_input, dec_source_mask, dec_target_mask):


        enc_embed = self.encoder_embedding.forward(enc_input)
        encoder_output = self.encoder_block.forward(enc_embed)

        dec_embed = self.decoder_embeddings.forward(dec_input)
        decoder_output = self.decoder_block.forward(dec_embed, encoder_output, dec_source_mask, dec_target_mask)

        model_output = self.output_gen.forward(decoder_output)

        return model_output