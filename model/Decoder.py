import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn import TransformerDecoder, TransformerDecoderLayer


from utils import *
from model import *

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

class TransDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.1, padding_idx=2):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.src_mask = None

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)

        decoder_layers = TransformerDecoderLayer(emb_dim, 8, hid_dim, dropout)
        decoder_norm = LayerNorm(emb_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layers, decoder_norm)

        # self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(2*emb_dim, output_dim, cutoffs=[1000,10000])

        self.dropout = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input, memory, has_mask=True):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        # input = input.unsqueeze(0)

        #input = [1, batch size]
        # hidden = hidden.view(self.n_layers, hidden.size(1), -1)

        if has_mask:
            device = input.device
            if self.src_mask is None or self.src_mask.size(0) != len(input):
                mask = self._generate_square_subsequent_mask(len(input)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        embedded = self.embedding(input) * math.sqrt(self.output_dim)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_decoder(embedded, memory, self.src_mask)

        return output

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.1, padding_idx=2):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=padding_idx)

        self.rnn = nn.LSTM(emb_dim+4*hid_dim, 2*hid_dim, n_layers, dropout = dropout)

        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(2*hid_dim, output_dim, cutoffs=[10,12])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, start_state):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]
        hidden = hidden.view(self.n_layers, hidden.size(1), -1)

        embedded = self.dropout(torch.cat((self.embedding(input), start_state), dim=2))



        #embedded = [1, batch size, emb dim]
        # print(hidden.size())
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]

        # prediction = self.out(output.squeeze(0))
        # print(output.size())
        prediction = self.adaptive_softmax.log_prob(output.squeeze(0))
        #prediction = [batch size, output dim]

        return prediction, hidden, cell
