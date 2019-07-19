import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn import TransformerDecoder, TransformerDecoderLayer


from utils import *
from model import *

class TransDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.1, padding_idx=2):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.src_mask = None

        decoder_layers = TransformerDecoderLayer(emb_dim, 8, hid_dim, dropout)
        decoder_norm = LayerNorm(emb_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layers, decoder_norm)

        # self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(2*emb_dim, output_dim, cutoffs=[1000,10000])

        self.dropout = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):


        device = tgt.device
        if tgt_mask is None or tgt_mask.size(0) != len(tgt):
            mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
            tgt_mask = mask
        else:
            tgt_mask = None

        output = self.transformer_decoder(tgt, memory, tgt_mask)

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

        self.rnn = nn.GRU(emb_dim, 2*hid_dim, n_layers, dropout=dropout)

        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(2*hid_dim, output_dim, cutoffs=[10,12])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, start_state):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]
        # hidden = hidden.view(self.n_layers, hidden.size(1), -1)
        embedded = self.embedding(input)

        # embedded = self.dropout(torch.cat((self.embedding(input), start_state), dim=2))



        #embedded = [1, batch size, emb dim]
        # print(hidden.size())
        output, hidden = self.rnn(embedded, hidden)

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

        return prediction, hidden
