import os
import torch
import torch.nn as nn
from transformers import BertModel

class BERT(nn.Module):
    def __init__(self, ag=False):
        super(BERT, self).__init__()
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'bert_model.pkl')
        if os.path.exists(model_path):
            self.bert = torch.load(model_path)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.linear = nn.Linear(768, 4 if ag else 2)


    def forward(self, inputs, attention_masks):
        bert_output = self.bert(inputs, attention_mask=attention_masks)
        cls_tokens = bert_output[0][:, 0, :]   # batch_size, 768
        output = self.linear(cls_tokens) # batch_size, 1(4)
        return output