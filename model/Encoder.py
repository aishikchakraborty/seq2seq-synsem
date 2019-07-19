import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from utils import *
import math


class TransEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.1, no_langs=3, pad_idx=2):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.src_mask = None

        self.M = nn.Parameter(torch.randn(hid_dim, hid_dim))
        self.N = nn.Parameter(torch.randn(no_langs, hid_dim, hid_dim))
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.n_layers = n_layers
        encoder_layers = TransformerEncoderLayer(emb_dim, 8, hid_dim, dropout)
        encoder_norm = LayerNorm(emb_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers, encoder_norm)

        self.dropout = nn.Dropout(dropout)



    def forward(self, src, mask=None, src_key_padding_mask=None):


        #embedded = [src sent len, batch size, emb dim]
        outputs = self.transformer_encoder(src)
        return outputs


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.1, no_langs=3, pad_idx=2):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.M = nn.Parameter(torch.randn(4*hid_dim, hid_dim))
        self.N = nn.Parameter(torch.randn(no_langs, 4*hid_dim, hid_dim))

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout)


        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src sent len, batch size]

        embedded = self.embedding(src)

        #embedded = [src sent len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)
        outputs = outputs.view(outputs.size(1), outputs.size(2), -1)

        # hidden =  hidden.view(hidden.size(1), -1)
        # hidden = F.max_pool1d(outputs, outputs.size(2)).squeeze(2)
        sem_emb = 0
        # sem_emb = torch.mm(hidden, self.M)
        # print(cell.size())
        # print(self.N[src_lang, :, :].size())
        # syn_emb = torch.bmm(hidden.squeeze(2).unsqueeze(1), self.N[src_lang, :, :]).squeeze(1)

        # hidden = torch.cat((sem_emb, syn_emb), dim=1).unsqueeze(0)

        fwd_final = hidden[0:hidden.size(0):2]
        bwd_final = hidden[1:hidden.size(0):2]
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer
        hidden = torch.cat([fwd_final, bwd_final], dim=2)

        return hidden, sem_emb

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters())
    #     return weight.new_zeros(self.n_layers*2, bsz, self.hid_dim)
