import pandas as pd
import re
import numpy as np
import pickle
import argparse
from collections import Counter
import statistics
import math


import torch
import torch.utils.data as data

PAD_INDEX = 0
UNK_INDEX = 1

def clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
#    string = re.sub(r"\'s", " \'s", string) 
#    string = re.sub(r"\'ve", " \'ve", string) 
#    string = re.sub(r"n\'t", " n\'t", string) 
#    string = re.sub(r"\'re", " \'re", string) 
#    string = re.sub(r"\'d", " \'d", string) 
#    string = re.sub(r"\'ll", " \'ll", string) 
#    string = re.sub(r",", "", string) 
#    string = re.sub(r"!", " ! ", string) 
#    string = re.sub(r"\(", " \( ", string) 
#    string = re.sub(r"\)", " \) ", string) 
#    string = re.sub(r"\?", " \? ", string) 
#    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX, "UNK": UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2  # Count default tokens
        self.word_num = 0

    def index_words(self, sentence):
        for word in sentence:
            self.word_num += 1
            
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words += 1
            else:
                self.word2count[word] += 1

def read_data(vocab, file_name):
    words_list = []
    revs = []
    with open(file_name, 'r', encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            rev = []
            rev.append(line.strip())
            orig_rev = clean(" ".join(rev))
            words_list += orig_rev.split()   
            revs.append({'content': orig_rev})
    return revs, words_list

def Lang(vocab, file_name):
    statistic = {"sent_num": 0, "word_num": 0, "vocab_size": 0,
                 "top_10_words": 0,
                 "max_len": 0, "avg_len": 0, "len_std": 0, "class_distribution": {}}
   
    vocab_size = {'UNK': 0}
    keys = {'content'}
    sentence = []
    
    revs, words_list = read_data(vocab, file_name)
    
    # number of sentences
    df = pd.read_csv(file_name)
    sent_num = len(df)
    statistic["sent_num"] = sent_num 
    
    # number of words
    word_num = len(words_list)
    statistic["word_num"] = word_num
    
    # number of unique words 
    for word in words_list:
        count = vocab_size.get(word, 0)
        count += 1
        vocab_size[word] = count
    statistic["vocab_size"] = len(vocab_size)
        
    # top 10 frequent words
    top_10_words = []
    c = Counter(words_list)
    top_10_words = c.most_common(10)
    statistic["top_10_words"] = top_10_words
        
    
    # Max sentence length
    for sentences in revs:
        sentence = {k: sentences[k] for k in keys}
        sentence = sentence.values()
        max_len = max(len(l) for l in sentence)
    statistic["max_len"] = max_len
    
    # average sentence length
    avg_len = word_num / sent_num
    statistic["avg_len"] = avg_len
    
    # Standard Deviation in number of words
    x_sqr = word_num
    x_ave = avg_len * avg_len
    len_std = math.sqrt(x_sqr - x_ave)
    statistic["len_std"] = float("{:.2f}".format(len_std))
    
    #Distribution of classes
    dis_0 = ((df["rating"] == 0) == True).sum()
    dis_1 = ((df["rating"] == 1) == True).sum()
    dis_2 = ((df["rating"] == 2) == True).sum()
    statistic["class_distribution"] = "0: {}, 1: {} 2: {}".format(dis_0, dis_1, dis_2)

    return vocab, statistic


class Dataset(data.Dataset):
    def __init__(self, data, vocab):
        self.id, self.X, self.y = data
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
        self.id = torch.LongTensor(self.id)
        if (self.y is not None): self.y = torch.LongTensor(self.y)

    def __getitem__(self, index):
        ind = self.id[index]
        X = self.tokenize(self.X[index])
        if (self.y is not None):
            y = self.y[index]
            return torch.LongTensor(X), y, ind
        else:
            return torch.LongTensor(X), ind

    def __len__(self):
        return self.num_total_seqs

    def tokenize(self, sentence):
        return [self.vocab.word2index[word] if word in self.vocab.word2index else UNK_INDEX for word in sentence]

def collate_fn(batch_size):
    label = torch.tensor([entry[0] for entry in batch_size])
    text = [entry[1] for entry in batch_size]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def preprocess(filename, max_len=200, test=False):
    df = pd.read_csv(filename)
    id_ = []  # review id
    rating = []  # rating
    content = []  # review content

    for i in range(len(df)):
        id_.append(int(df['id'][i]))
        if not test:
            rating.append(int(df['rating'][i]))
        sentence = clean(str(df['content'][i]).strip())
        sentence = sentence.split()
        sent_len = len(sentence)
        # here we pad the sequence for whole training set, you can also try to do dynamic padding for each batch by customize collate_fn function
        # if you do dynamic padding and report it, we will give 1 points bonus
        if sent_len > max_len:
            content.append(sentence[:max_len])
        else:
            content.append(sentence + ["PAD"] * (max_len - sent_len))

    if test:
        len(id_) == len(content)
        return (id_, content, None)
    else:
        assert len(id_) == len(content) == len(rating)
        return (id_, content, rating)


def get_dataloaders(batch_size, max_len):
    vocab = Vocab()
    vocab, statistic = Lang(vocab, "train.csv")
    print(statistic)
    
#    max_len = collate_fn()
    
    train_data = preprocess("train.csv", max_len)
    dev_data = preprocess("dev.csv", max_len)
    test_data = preprocess("test.csv", max_len, test=True)
    train = Dataset(train_data, vocab)
    dev = Dataset(dev_data, vocab)
    test = Dataset(test_data, vocab)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev, batch_size=batch_size, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    return data_loader_tr, data_loader_dev, data_loader_test, statistic["vocab_size"]

