import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        self.embd = nn.Embedding(args.max_len, args.kernel_num)
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[0]), stride=1),
            nn.ReLU(), nn.MaxPool1d(kernel_size=int(args.kernel_sizes[0]), stride=3))
                        # nn.AveragePooling1d(kernel_size=int(args.kernel_sizes[0]), stride=3)
        self.conv2 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[2]), stride=1),
            nn.ReLU(),nn.MaxPool1d(kernel_size=int(args.kernel_sizes[2]), stride=3)) 
                        # nn.AveragePooling1d(kernel_size=int(args.kernel_sizes[2]), stride=3)
        self.conv3 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[4]), stride=1),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[4]), stride=1),
            nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(1200, 1200), nn.ReLU(), nn.Dropout(p=args.dropout))
        self.fc2 = nn.Sequential(nn.Linear(1200, 1200), nn.ReLU(), nn.Dropout(p=args.dropout))
        self.fc3 = nn.Linear(1200, args.class_num)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.embd(x)
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        return x

class CharCNN(nn.Module):
    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        self.embd = nn.Embedding(args.max_len, args.kernel_num)
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[0]), stride=1),
            nn.ReLU(),nn.MaxPool1d(kernel_size=int(args.kernel_sizes[0]), stride=3))
                    # nn.AveragePooling1d(kernel_size=int(args.kernel_sizes[0]), stride=3)
        self.conv2 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[2]), stride=1),
            nn.ReLU(), nn.MaxPool1d(kernel_size=int(args.kernel_sizes[2]), stride=3))            
                    # nn.AveragePooling1d(kernel_size=int(args.kernel_sizes[2]), stride=3)
        self.conv3 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[4]), stride=1),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv1d(args.kernel_num, args.embed_dim, kernel_size=int(args.kernel_sizes[4]), stride=1),
            nn.ReLU())
        
        self.fc1 = nn.Sequential(nn.Linear(1200, 1200), nn.ReLU(), nn.Dropout(p=args.dropout))
        self.fc2 = nn.Sequential(nn.Linear(1200, 1200), nn.ReLU(), nn.Dropout(p=args.dropout))
        self.fc3 = nn.Linear(1200, args.class_num)
        self.log_softmax = nn.LogSoftmax()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
