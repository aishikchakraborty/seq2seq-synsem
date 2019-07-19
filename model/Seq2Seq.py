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

class TransSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, emb_dim, nout, cutoffs=[10, 11]):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.embed_dim = emb_dim
        # self.decoder.embedding.weight = self.encoder.embedding.weight
        # self.transformer = nn.Transformer(custom_encoder=encoder, custom_decoder=decoder)

        self.proj = nn.Linear(emb_dim, nout)


    def forward(self, src, trg, tgtlng, teacher_forcing_ratio = 0.8):

        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        memory = self.encoder(src)
        outputs = self.decoder(trg, memory)
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
