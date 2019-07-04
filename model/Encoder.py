import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.5, no_langs=3):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.M = nn.Parameter(torch.randn(hid_dim, hid_dim//2))
        self.N = nn.Parameter(torch.randn(no_langs, hid_dim, hid_dim//2))

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lang):

        #src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        sem_emb = torch.mm(hidden.squeeze(0), self.M)
        # print(self.N[src_lang, :, :].size())
        syn_emb = torch.bmm(hidden.squeeze(0).unsqueeze(1), self.N[src_lang, :, :]).squeeze(1)

        hidden = torch.cat((sem_emb, syn_emb), dim=1).unsqueeze(0)

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell
