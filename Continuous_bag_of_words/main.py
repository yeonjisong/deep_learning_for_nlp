#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:55:02 2020

@author: yeonjisong



"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
import torch.nn.functional as F
import io
import re
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords



from CBOW import CBOW
import loader

CONTEXT_SIZE = 2
EMBEDDING_DIM = 30
EPOCH = 100
VERVOSE = 5

def train_cbow(data, unique_vocab, word_to_idx):
    cbow = CBOW(len(unique_vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    nll_loss = nn.NLLLoss()  # loss function
    optimizer = SGD(cbow.parameters(), lr=0.001)
    
    print("data length: ", len(data))
    
    for epoch in range(EPOCH):
        total_loss = 0

        for context, target in data:
            try:
                inp_var = torch.tensor([word_to_idx[word] for word in context], dtype=torch.long)
                target_var = torch.tensor([word_to_idx[target]], dtype=torch.long)
    
                cbow.zero_grad()
                log_prob = cbow(inp_var)
                loss = nll_loss(log_prob, target_var)
                loss.backward()
                optimizer.step()
                total_loss += loss.data
            except KeyError:
                pass
        
        if epoch % VERVOSE == 0:
            loss_avg = float(total_loss / len(data))
            print("{}/{} loss {:.2f}".format(epoch, EPOCH, loss_avg))
    return cbow

def main():
    dataset = loader.MyData()
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-fn','--file_name', help='file name', required=False, default='myTest')
    args = vars(parser.parse_args())
    print(args)

    data = list()
    src = dataset.__getline__()
    
    for i in range(CONTEXT_SIZE, len(src) - CONTEXT_SIZE):
        data_context = list()
        for j in range(CONTEXT_SIZE):
            data_context.append(src[i - CONTEXT_SIZE + j])
        
        for j in range(1, CONTEXT_SIZE + 1):
            data_context.append(src[i + j])
      
        data_target = src[i]
        data.append((data_context, data_target))
 
#    print("Some data: ", data[:10])
    
    unique_vocab = dataset.__getvocab__()
    word_to_idx = {w:i for i, w in enumerate(unique_vocab)}
#    print('word_to_idx: ', word_to_idx)

    train_cbow(data, unique_vocab, word_to_idx)

if __name__ == "__main__":
    main()
