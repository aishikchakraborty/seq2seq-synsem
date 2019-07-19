import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import *
from model import *

def detach_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

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

class TransSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, emb_dim, nout, cutoffs=[10, 11], pad_idx=2):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.embed_dim = emb_dim
        self.input_dim = nout
        self.embedding = nn.Embedding(self.input_dim, emb_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(emb_dim, 0.1)
        # self.decoder.embedding.weight = self.encoder.embedding.weight


        self.transformer = nn.Transformer(d_model=emb_dim, custom_encoder=encoder, custom_decoder=decoder)

        self.proj = nn.Linear(emb_dim, nout)


    def forward(self, src, tgt, tgtlng, teacher_forcing_ratio = 1.0):

        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        src = self.embedding(src) * math.sqrt(self.input_dim)
        src = self.pos_encoder(src)

        tgt = self.embedding(tgt) * math.sqrt(self.input_dim)
        tgt = self.pos_encoder(tgt)

        outputs = self.transformer(src, tgt)
        outputs = F.log_softmax(self.proj(outputs), dim=-1)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, nout, cutoffs=[10, 12]):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.decoder.embedding.weight = self.encoder.embedding.weight


        # self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(nout, nout, cutoffs=cutoffs)

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, tgtlng, teacher_forcing_ratio = 1.0):

        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, sem_emb = self.encoder(src)
        # syn_emb = torch.bmm(hidden.unsqueeze(1), self.encoder.N[tgtlng, :, :]).squeeze(1)


        # start_state = torch.cat((sem_emb, syn_emb), dim=1).unsqueeze(0)
        start_state = hidden
        # hidden = start_state
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(max_len):
            output, hidden = self.decoder(input, hidden, start_state)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
            detach_hidden(hidden)

        return outputs
