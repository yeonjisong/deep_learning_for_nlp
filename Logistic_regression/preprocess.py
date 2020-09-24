import numpy as np
import re
from collections import Counter
import io



def read_data(_file, cleaning):
    revs = []
    max_len = 0
    words_list = []
    with io.open(_file, "r",  encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            ID, label, sentence = line.split('\t')
            label_idx = 1 if label=='pos' else 0 # 1 for pos and 0 for neg
            rev = []
            rev.append(sentence.strip())
            print(rev[0])
            if cleaning:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev)
            
            revs.append({'y':label_idx, 'txt':orig_rev})
            
            words_list += orig_rev.split()

    return revs, words_list

def clean_str(string):
    
    string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r'\W',' ',string)
    string = re.sub(r'\s+',' ',string)
    string = re.sub("\'ve", " \'ve", string)
    string = re.sub("n\'t", " n\'t", string)
    string = re.sub("\'re", " \'re", string)
    string = re.sub("\'d", " \'d", string)
    string = re.sub("\'ll", " \'ll", string)
    string = re.sub(",", " , ", string)
    string = re.sub("!", " ! ", string)
    string = re.sub("\(", r" \( ", string)
    string = re.sub("\)", r" \) ", string)
    string = re.sub("\?", r" \? ", string)
    string = re.sub("\s{2,}", " ", string)
    string = re.sub("[!,]", " ", string)
    string = re.sub("[@]", " ", string)
    
    return string.strip().lower()


def build_vocab(words_list, max_vocab_size=-1):     
    
    # Number of vocabulary
    word2idx = {'UNK': 0} # UNK is for unknown word
    if max_vocab_size == -1:
        for word in words_list:
            count = word2idx.get(word, 0)
            count += 1
            word2idx[word] = count
     
    # Top 10 words
    top_10_words = []
    c = Counter(words_list)
    top_10_words = c.most_common(10)
    
        
    return word2idx, top_10_words

def get_info(revs, words_list):

    nb_sent, max_len, word_count = 0, 0, 0
    sentence = []
    keys = {'txt'}
    
    # Numb of sentences
    nb_sent = len(revs)
    
    #Numb of words
    word_count = len(words_list)

    # Max sentence length
    for sentences in revs:
        sentence = {k: sentences[k] for k in keys}
        sentence = sentence.values()
        max_len = max(len(l) for l in sentence)

    return nb_sent, max_len, word_count

def data_preprocess(_file, cleaning, max_vocab_size):
    revs, words_list = read_data(_file, cleaning)
    nb_sent, max_len, word_count = get_info(revs, words_list)
    word2idx, top_10_words = build_vocab(words_list, max_vocab_size) 
    # data analysis
    print("Number of words: ", word_count)
    print("Max sentence length: ", max_len)
    print("Number of sentences: ", nb_sent)
    print("Number of vocabulary: ", len(word2idx))
    print("Top 10 most frequently words", top_10_words)

    return revs, word2idx

def feature_extraction_bow(revs, word2idx):
    """
        data is a 2-D array with the size (nb_sentence*nb_vocab)
        label is a 2-D array with the size (nb_sentence*1)
    """
    data = []
    label = []
    testlabel = []
    testdata = []
    BOV=[]
    keys = {'txt'}
    
    for sent_info in revs:
        label = sorted(list(set(label)))
        label = {k: sent_info[k] for k in keys}
        label = label.values()
        label = ''.join(label)        
        data = label.split()
        label = np.reshape(label, 1)
        bov = np.zeros(len(word2idx))    
        testlabel.append(label)
        testdata.append(data)

        for w in data:
            for i,word in enumerate(word2idx):
                    if word == w:
                        bov[i] += 1
        BOV.append(bov)            
#        print("{0}\n{1}\n".format(label,np.array(bov)))
#    label = [list(i) for i in testlabel]
#    data = [list(i) for i in testdata]
#    data = np.reshape(len(revs),BOV)
    data = np.array(BOV)
    label = BOV
    return np.array(data), np.array(label)

def normalization(data):    
#    print(data)
#    to mean=0, std=1
#    data = np.array(data, dtype=float)
#    data = data.ast1ype(np.float)
    data -= np.mean(data, axis = 0)
#    data -= np.expand_dims(np.mean(data, axis = 1),1)
#    data /= np.std(data, axis = 0)
#    print("std sucecss!!!!!!!")

    return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f','--file', help='input csv file', required=False, default='./twitter-sentiment-testset.csv')
    parser.add_argument('-c','--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv','--max_vocab', help='max vocab size predifined, no limit if set -1', required=False, default=-1)
    args = vars(parser.parse_args())
    print(args)

    revs, word2idx = data_preprocess(args['file'], args['clean'], int(args['max_vocab']))

    #Assignment1
    data, label = feature_extraction_bow(revs, word2idx)
    data = normalization(data)

    
    
    
    
    
    
    
    