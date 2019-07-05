import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.5, no_langs=3, pad_idx=2):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.M = nn.Parameter(torch.randn(2*hid_dim, hid_dim))
        self.N = nn.Parameter(torch.randn(no_langs, 2*hid_dim, hid_dim))
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.n_layers = n_layers
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = outputs.view(outputs.size(1), outputs.size(2), -1)

        hidden =  hidden.view(hidden.size(1), -1)
        hidden = F.max_pool1d(outputs, outputs.size(2)).squeeze(2)

        sem_emb = torch.mm(hidden, self.M)
        # print(cell.size())
        # print(self.N[src_lang, :, :].size())
        # syn_emb = torch.bmm(hidden.squeeze(2).unsqueeze(1), self.N[src_lang, :, :]).squeeze(1)

        # hidden = torch.cat((sem_emb, syn_emb), dim=1).unsqueeze(0)

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell.view(self.n_layers, cell.size(1), -1), sem_emb
