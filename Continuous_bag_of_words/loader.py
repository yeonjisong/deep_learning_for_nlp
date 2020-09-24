#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:39:09 2020

@author: yeonjisong
"""
from torch.utils.data import Dataset, DataLoader
import gzip
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
import re


class MyData(Dataset):
    unique_yy = []
    def __init__ (self, filename = "./reviews_data.txt.gz"):
        self.len = 0
        with gzip.open(filename, 'rt', encoding="ISO-8859-1") as f:

            self.filename = filename
            self.targetLines = [x.strip() for x in f if x.strip()]
            self.srcLines = [x.lower().replace('\t', ' ') for x in self.targetLines]
            self.srcLines = [re.sub('[^A-Za-z0-9]+', ' ', x) for x in self.srcLines]
            vocab = ''.join(str(e) for e in self.srcLines[:10])
            self.unique_vocab = [word for word in word_tokenize(vocab) if not word in stopwords.words()]
            self.len = len(self.srcLines)
            print('sentence extracted from file:', self.len, 'sentence')
            print('Dictionary built: ', len(self.unique_vocab), 'words')
            
    def __getitem__ (self, index):
        return self.srcLines[index], self.targetLines[index]
    
    def __getline__ (self):
        self.srcLines = ''.join(str(e) for e in self.srcLines[:10])
        self.srcLines = self.srcLines.split(' ')
        return self.srcLines
    
    def __getvocab__ (self):
        return self.unique_vocab
        
    def __len__ (self):
        return self.len

if __name__ == "__main__":
    dataset = MyData()
    bsz = 32
    train_loader = DataLoader(dataset=dataset, batch_size=bsz, shuffle = True)
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f','--file', help='input csv file', required=False, default='./twitter-sentiment-testset.csv')
    parser.add_argument('-c','--clean', help='True to do data cleaning, default is False', action='store_false')
    args = vars(parser.parse_args())
    print(args)

#    for i, (src, target) in enumerate(train_loader):
##        print(i, "data", target)
#        print(i, "data", src)