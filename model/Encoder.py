import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from utils import *
import math

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.1, no_langs=3, pad_idx=2):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        self.M = nn.Parameter(torch.randn(hid_dim, hid_dim))
        self.N = nn.Parameter(torch.randn(no_langs, hid_dim, hid_dim))
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.n_layers = n_layers
        encoder_layers = TransformerEncoderLayer(emb_dim, 8, hid_dim, dropout)
        encoder_norm = LayerNorm(emb_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers, encoder_norm)

        self.dropout = nn.Dropout(dropout)



    def forward(self, src, has_mask=True):
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        #src = [src sent len, batch size]

        embedded = self.embedding(src) * math.sqrt(self.input_dim)
        embedded = self.pos_encoder(embedded)

        #embedded = [src sent len, batch size, emb dim]
        outputs = self.transformer_encoder(embedded)
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
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = outputs.view(outputs.size(1), outputs.size(2), -1)

        hidden =  hidden.view(hidden.size(1), -1)
        # hidden = F.max_pool1d(outputs, outputs.size(2)).squeeze(2)
        sem_emb = 0
        # sem_emb = torch.mm(hidden, self.M)
        # print(cell.size())
        # print(self.N[src_lang, :, :].size())
        # syn_emb = torch.bmm(hidden.squeeze(2).unsqueeze(1), self.N[src_lang, :, :]).squeeze(1)

        # hidden = torch.cat((sem_emb, syn_emb), dim=1).unsqueeze(0)

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell.view(self.n_layers, cell.size(1), -1), sem_emb
