import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class ProbingTasks(nn.Module):
    def __init__(self, encoder, hid_dim, num_classes_sentlen, num_classes_wordpair, requires_grad = False):
        super().__init__()

        self.encoder = encoder
        self.embedding = self.encoder.embedding
        self.l1 = nn.Linear(hid_dim, num_classes_sentlen)
        self.l2 = nn.Linear(hid_dim + 2*self.embedding.weight.size(1), num_classes_wordpair)

        for p in self.encoder.parameters():
            p.requires_grad = requires_grad
        self.embedding.weight.requires_grad = requires_grad


    def forward(self, src, srclng, wordpairs):

        hidden, cell, sem_emb = self.encoder(src)
        syn_emb = torch.bmm(hidden.unsqueeze(1), self.encoder.N[srclng, :, :]).squeeze(1)

        sentlen_out  = self.l1(syn_emb)
        emb_word_pairs = self.embedding(wordpairs).view(src.size(1), -1)
        wp_out  = self.l2(torch.cat((syn_emb, emb_word_pairs), dim=1))

        return sentlen_out, wp_out
