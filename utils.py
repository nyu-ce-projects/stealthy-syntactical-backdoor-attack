import torch
from torchtext import vocab as Vocab
import os
import collections
import pandas as pd

def get_vocab(target_set):
    tokenized_data = [[word.lower() for word in data_tuple[0].split(' ')] for data_tuple in target_set]
    counter = collections.Counter([word for review in tokenized_data for word in review])
    vocab = Vocab.Vocab(counter, min_freq=5)
    return vocab

def read_data(data_path,data_type):
    file_path = os.path.join(data_path, data_type+'.tsv')

    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

def write_data(path,filename,data):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)

        
def get_device():
    n_gpus = 1
    if torch.cuda.is_available():
        device = 'cuda' 
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device,n_gpus


from contextlib import contextmanager

@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request

    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})
    
def write_data(path,data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)
