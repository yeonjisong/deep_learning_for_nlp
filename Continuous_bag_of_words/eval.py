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
import loader_eval

CONTEXT_SIZE = 2
EMBEDDING_DIM = 30
EPOCH = 100
VERVOSE = 5


def train_cbow(data, unique_vocab, word_to_idx):
    cbow = CBOW(len(unique_vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    nll_loss = nn.NLLLoss()  # loss function
    optimizer = SGD(cbow.parameters(), lr=0.001) # optimizer
    
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

def test_cbow(cbow, unique_vocab, word_to_idx):
    word_1 = unique_vocab[286]
    word_2 = unique_vocab[543]
    
    # compare a pair of  words
    print("First word: ", word_1)
    print("Second word: ", word_2)
    word_1_vec = cbow.get_word_vector(word_to_idx[word_1])[0]
    word_2_vec = cbow.get_word_vector(word_to_idx[word_2])[0]
    
    # Similarity between the chosen pair
    word_similarity = (word_1_vec.dot(word_2_vec) / (torch.norm(word_1_vec) * torch.norm(word_2_vec))).data.numpy()
    print("Similarity between '{}' & '{}' : {:0.4f}".format(word_1, word_2, word_similarity))

def matrix(cbow, unique_vocab, word_to_idx):
    for i in range(len(unique_vocab)):
        word = unique_vocab[i]
        matrix = cbow.get_word_vector(word_to_idx[word])[0]
    return matrix

def sigmoid(z):
    s = 1 / (1 + np.exp(np.dot(-1, z)))
    return s

def predict(X):
    m = len(X)
    Y_prediction = np.zeros((1, m))
    X = torch.tensor(X, dtype=torch.int8)
    for i in X:
        if X[i] > 0:
            Y_prediction[0, i] = 1
        else:
            0
        pass
    return Y_prediction

def write_testset_prediction(test_data, file_name="myPrediction.csv"):
    Y_prediction_test = predict(test_data)
    f_pred = open(file_name, 'w')
    f_pred.write('ID\tSentiment')
    ID = 0
    for pred in Y_prediction_test[0]:
        sentiment_pred = 'pos' if pred==0 else 'neg'
        f_pred.write(str(ID)+','+sentiment_pred+'\n')
        ID += 1
        
def main():
    dataset = loader_eval.MyData()
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-fn','--file_name', help='file name', required=False, default='sample-submission')
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
    
    unique_vocab = dataset.__getvocab__()
    word_to_idx = {w:i for i, w in enumerate(unique_vocab)}

    cbow = train_cbow(data, unique_vocab, word_to_idx)
    
    # Word similarity
    test_cbow(cbow, unique_vocab, word_to_idx)
    print('\n[Start evaluating: ]'.format(args['file_name']+'.csv'))
    revs = matrix(cbow, unique_vocab, word_to_idx)
    write_testset_prediction(revs, args['file_name']+'.csv' )

    

    
if __name__ == "__main__":
    main()
