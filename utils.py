from torchtext import vocab as Vocab
import os
import collections
import pandas as pd

def get_vocab(target_set):
    tokenized_data = [[word.lower() for word in data_tuple[0].split(' ')] for data_tuple in target_set]
    counter = collections.Counter([word for review in tokenized_data for word in review])
    vocab = Vocab.Vocab(counter, min_freq=5)
    return vocab

def read_data(base_path,data_type,poisoned):
    if poisoned:
        data_purity = "poison"
    else:
        data_purity = "clean"
    file_path = os.path.join(base_path,data_purity, data_type+'.tsv')

    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data
    
def write_data(path,data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)
