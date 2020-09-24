#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:52:07 2020

@author: yeonjisong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super(CBOW, self).__init__()
 
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
 
        self.lin1 = nn.Linear(self.context_size * 2 * self.embedding_size, 512)
        self.lin2 = nn.Linear(512, self.vocab_size)
    
    def forward(self, inp):
        out = self.embeddings(inp).view(1, -1)
        out = out.view(1, -1)
        out = self.lin1(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.log_softmax(out, dim=1)
        return out
    
    def __embedding_mat__(self):
        return nn.Embedding(self.vocab_size, self.embedding_size)
    
    def get_word_vector(self, word_idx):
        try:
            word = Variable(torch.LongTensor([word_idx]))
        except KeyError:
            pass
        return self.embeddings(word).view(1, -1)
    
    def __len__(self):
        return self.vocab_size
    
if __name__ == "__main___":
    print("cbow.py success")
    cbow_model = CBOW()
