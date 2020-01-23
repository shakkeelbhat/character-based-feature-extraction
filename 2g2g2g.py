import os
import argparse
import copy
import math
import json
from time import time
import logging, pickle
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import codecs


class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        self.args = args
        vocab_size = args.vocab_size
        h_model= args.h_model
        d_model = args.d_model
        #word_dim gives the size of char embeddings
        char_dim = args.char_len
        # hidden_dim gives the dimension of the number of hidden layers in char,word and sentence GRUs
        hidden_dim = args.hidden_dim
        #doc_len is the maximum number of sentences; sent_len is the maximum numebr of words in a sentence
        doc_len = args.doc_len
        sent_len = args.sent_len
        word_len=args.word_len


        # Define the Embedding layer for the chars
        self.emb_layer = nn.Embedding(vocab_size, char_dim)
        
        self.char_conv=[nn.Conv2d(1,256,[i,15]) for i in (7,7,3,3,3,3)] 
        self.char_pool=[nn.MaxPool2d([22+i,1]) for i in (0,0,4,4,4,4)]
        # Pass the char Embeddings through the GRU
        self.char_GRU = nn.GRU(char_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        
        
        
        #define layers for words
        self.word_conv=[nn.Conv2d(1,256,[i,15]) for i in (7,7,3,3,3,3)] 
        self.word_pool=[nn.MaxPool2d([22+i,1]) for i in (0,0,4,4,4,4)]
        # Pass the word Embeddings through the GRU
        self.word_GRU = nn.GRU(word_dim, hidden_dim, 1, bidirectional=True, batch_first=True)

       
        
        #define layers for sentences
        self.word_conv=[nn.Conv2d(1,256,[i,15]) for i in (7,7,3,3,3,3)] 
        self.sent_pool=[nn.MaxPool2d([22+i,1]) for i in (0,0,4,4,4,4)]
        # Pass the word Embeddings through the GRU
        self.sent_GRU = nn.GRU(d_model, hidden_dim, 1, bidirectional=True, batch_first=True)

       
        # Compute the abstract features
        self.doc_linear = nn.Linear(d_model, d_model)
        


    def forward(self, x):
        # ndocs is the batch size, i.e., number of documents in a batch
            ndocs = x.size(0)
            doc_len = x.size(1)
            sent_len = x.size(2)
            word_len= x.size(3)

        # x will have shape (ndocs, doc_len, sent_len,word_len)

        # Get the embeddings of the words; embeddings will be of shape (ndocs, doc_len, sent_len, word_len, emb_dim)
            x = self.emb_layer(x)
            char_dim = x.size(-1)
            x = x.reshape((-1,1, word_len,char_dim))
            #x = char_conv(x)
            #x = char_pool(x).squeeze()
            #print('hereee',x.shape)
#             print(type(x))
            x = [self.char_pool[j](self.char_conv[j](x)).squeeze() for j in range(6)]#char_rep
            
            print('hereee',x.shape)
            x = torch.cat(x,dim=0)
            
            x = x.reshape(-1, doc_len, d_model)
            x, _ = self.char_GRU(x)
        

            x = self.char_pool(x.reshape(-1,1,1, word_len))
           #Pass through the word GRU. It expects input in the form (batch, seq_len, input_size)
           # x = word_conv(x)
            #x = word_pool_pool(x).squeeze()
            x = [self.word_pool[j](self.word_conv[j](x)).squeeze() for j in range(len(word_pool))]#word_rep
            x = torch.cat(x,dim=0)
            x = x.reshape(-1, doc_len, d_model)
            x,_ = self.word_GRU(x)
            x = self.word_pool(x.reshape(-1,1,sent_len, 1))
            #Pass through the sentence GRU. It expects input in the form (batch, seq_len, input_size)
            #x = sent_conv(x)
            #x = sent_pool(x).squeeze()
            x = [self.sent_pool[j](self.sent_conv[j](x)).squeeze() for j in range(len(sent_pool))]#sent_rep
            x = torch.cat(x,dim=0)
            x = x.reshape(-1, doc_len, d_model)
            x,_ = self.sent_GRU(x)
        
            #Average pool to get Document representation
            doc_rep = self.sent_pool(x.reshape(-1,1,doc_len, 1))
        
            #Pass the doc_rep through a linear layer and tanh non-linearity
            doc_rep = torch.softmax(self.doc_linear(doc_rep))
            



def getDocumentVector(path, doc_len, sent_len, word_len, char2id):
    X=np.zeros((batch_size,Nsent,Nword,Nchar), dtype=np.long)
    with open(path, encoding='utf-8')as f:
        doc = f.read().split('\n')[:doc_len]
        #print('doc',doc)
    for sent_no, line in enumerate(doc):
        words = line.split(' ')[:sent_len]
        #print('words',words)
        for word_no, word in enumerate(words):
            tokens=[i for i in word][:word_len]
            #print('tokens',tokens)
            for char_no,char in enumerate(tokens):
                #print('here ',sent_no,word_no,char_no)
                try:
                    X[sent_no,word_no,char_no] = char2id[char]
                except KeyError:
                    X[sent_no,word_no,char_no] = char2id['UNK']
    
    return X



import torch
from torch.utils import data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, path, doc_len, sent_len, word_len, char2id):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.path = path
        self.doc_len = doc_len
        self.sent_len = sent_len
        self.word_len = word_len
        self.char2id = char2id

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        path = os.path.join(self.path, ID)
        X = getDocumentVector(path, self.doc_len, self.sent_len, self.word_len, self.char2id)
        y = self.labels[ID]
        y = y[:doc_len]
        y.extend([0 for _ in range(len(y),doc_len)])
        y = list(map(float, y))
        y = np.array(y)

        return X, y


import json
with open('/home/shakeel/cnn2/char2id.json') as f:
    char2id = json.load(f)



class arguments():
    def __init__(self, **kwargs):
        self.vocab_size = kwargs['vocab_size']
        self.char_len = kwargs['char_len']
        self.hidden_dim = kwargs['hidden_dim']
        self.sent_len = kwargs['sent_len']
        self.word_len = kwargs['word_len']
        self.device = kwargs['device']
        self.ndocs = kwargs['ndocs']
        self.h_model=kwargs['h_model']
        self.d_model=kwargs['d_model']
        self.doc_len = kwargs['doc_len']

vocab_size, char_len = 63,10
word_len = 50
sent_len =100
word_len=15
doc_len=50
ndocs   = 16
h_model, d_model = 8, 512
hidden_dim = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
args_dict = {
    'vocab_size':vocab_size,
    'char_len':char_len,
    'ndocs':ndocs,
    'doc_len':doc_len,
    'word_len':word_len,
    'sent_len':sent_len,
    'hidden_dim':hidden_dim,
    'device':device,
    'h_model':h_model,
    'd_model':d_model,
}
args = arguments(**args_dict)


import pickle
with open('/home/shakeel/cnn2/args.json','wb') as f:
    pickle.dump(args, f)
with open('/home/shakeel/cnn2/args_dict.json','wb') as f:
    pickle.dump(args_dict, f)


def eval_network(network, data_iter, criterion, device):
    network.eval()
    total_loss = 0
    batch_num = 0
    accuracy = 0.0
#     iteration=3
    for features, targets in data_iter:
        features,targets = Variable(features.type(torch.long)).to(device), Variable(targets.float()).to(device)
        probs = network(features)
        loss = criterion(probs,targets)
        y_pred = np.round(probs.clone().detach().numpy())
        y_true = targets.clone().detach().numpy()
#         print(y_true.shape, y_pred.shape, probs)
        accuracy += accuracy_score(y_true,y_pred )
        total_loss += loss
        batch_num += 1
#         #PLEASE REMOVE THE NEXT LINE
#         iteration -= 1
#         if iteration == 1: break
    loss = total_loss / batch_num
    accuracy /= batch_num
    network.train()
    return loss, accuracy


def train(args, epochs=1, previous_model=False):
 
    with open('/home/shakeel/cnn/partition.json',encoding='utf-8') as f:
        partition = json.load(f)
    with open('/home/shakeel/cnn/labels.json',encoding='utf-8') as f:
        labels = json.load(f)
        
    
    #Step3: Create the train and validation datasets
    train_dataset = Dataset(partition['train'], labels['train'], '/home/shakeel/cnn/data/train', 
                            args.doc_len, args.sent_len, args.word_len, char2id)
    valid_dataset = Dataset(partition['valid'], labels['valid'], '/home/shakeel/cnn/data/valid/', 
                            args.doc_len, args.sent_len, args.word_len, char2id)
    
    #Step 4: Create the dataloaders
    train_dataloader = data.DataLoader(train_dataset, args.ndocs, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, args.ndocs, shuffle=False)
    
    # Step 5: Create the network
  
    network = net(args).to(args.device)

    # Step 6: Loss function
    criterion = nn.BCELoss()
    
    # Step 7: model info
    print(network)
    
    params = sum(p.numel() for p in list(network.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    
    #Step 8: Create the optimizer
    optimizer = torch.optim.Adam(network.parameters(),lr=1e-3)
    
    
    # Step 9: Early Stopping details
    min_val_loss = float('inf')
    n_epochs_stop = 5
    epochs_no_improve = 0
    network.cuda()
    network.train()
    type(type(network))
    
    t1 = time()
    #Step 10: Start traing for epochs
    for epoch_id in range(epochs):
        for i,(x,y) in enumerate(train_dataloader):
            print(args.device)
            x = Variable(x.type(torch.long)).to(device)
            y = Variable(y.float()).to(device)
            print(type(x), type(y))
            probs = network(x).to(device)
            loss = criterion(probs,y)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            if i%10==0:
                print('Epoch %d, Batch ID:%d Loss:%f' %(epoch_id, i,loss))
            
        val_loss, acc = eval_network(network, valid_dataloader, criterion, args.device)
        print('Epoch %d Loss:%f Accuracy:%f' %(epoch_id,loss, acc))
        # If the validation loss is at a minimum
        if val_loss < min_val_loss:
            # Save the model
            torch.save({
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, '/home/shakeel/cnn/checkpoints/best_model.pt')
#             torch.save(network, './checkpoints/best_model.pt')
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                # Load in the best model
                network = torch.load('/home/shakeel/cnn/checkpoints/best_model.pt')

    t2 = time()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))



batch_size = 64 # No. of documents in a batch
Nsent=100
Nword=50
Nchar=15
torch.cuda.set_device(1)
train(args, True)
