import os
import torch
import torch.nn as nn
from transformers import BertModel,BertForSequenceClassification

class BERT(nn.Module):
    def __init__(self, ag=False):
        super(BERT, self).__init__()
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'bert_model.pkl')
        if os.path.exists(model_path):
            self.bert = torch.load(model_path)
        else:
            num_labels = 4 if ag else 2
            self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_labels)


    def forward(self, inputs, attention_masks):
        bert_output = self.bert(inputs, attention_mask=attention_masks)
        cls_tokens = bert_output[0]
        return cls_tokens