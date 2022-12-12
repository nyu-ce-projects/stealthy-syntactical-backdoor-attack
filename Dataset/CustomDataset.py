import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils import get_vocab,read_data
from torch.nn.utils.rnn import pad_sequence

DEFAULT_DATA_PATH = './data/olid/'

class CustomDataset(Dataset):
    def __init__(self, data_type, data_purity) -> None:
        super(CustomDataset).__init__()
        self.data_type = data_type
        self.data_purity = data_purity
        
    def get_tokenized_data(self,data_path=DEFAULT_DATA_PATH):
        self.data = read_data(data_path,self.data_type,self.data_purity)
        if self.data_type=='train':
            train_set = self.data
        else:
            train_set = read_data(data_path,'train','poison')
        vocab = get_vocab(train_set)
        self.tokenized_data = [[vocab.stoi[word.lower()] for word in data_tuple[0].split(' ')] for data_tuple in self.data]
        self.labels = [data_tuple[1] for data_tuple in self.data]
        assert len(self.labels) == len(self.tokenized_data)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.tokenized_data[index], self.labels[index]

    def fn(self,data):
        labels = torch.tensor([item[1] for item in data])
        lengths = [len(item[0]) for item in data]
        texts = [torch.tensor(item[0]) for item in data]
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        return padded_texts, lengths, labels



class CustomDatasetForBert(Dataset):
    def __init__(self,data_type, data_purity) -> None:
        super(CustomDatasetForBert).__init__()
        self.data_type = data_type
        self.data_purity = data_purity


    def get_tokenized_data(self,data_path):
        self.data = read_data(data_path,self.data_type,self.data_purity)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenized_data = []
        self.labels = []
        for text, label in self.data:
            self.tokenized_data.append(torch.tensor(tokenizer.encode(text)))
            self.labels.append(label)
        assert len(self.tokenized_data) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.tokenized_data[index], self.labels[index]

    def fn(self, data):
        texts = []
        labels = []
        for text, label in data:
            texts.append(text)
            labels.append(label)
        labels = torch.tensor(labels)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
        return padded_texts, attention_masks, labels
