import OpenAttack
import tqdm
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import *
from Models.BERT import BERT
from DataPoisoning.SCPNPoisoning import SCPNPoisoning

from transformers import BertTokenizer
# tokenizer = PunctTokenizer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MyClassifier(OpenAttack.Classifier):
    def _init_(self, model) -> None:
        self.model = model
        # self.vocab = vocab
    
    def get_prob(self, sentences):
        # print(sentences)
        with torch.no_grad():
            texts = [torch.tensor(tokenizer.encode(sent)) for sent in sentences]
            padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
            attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
            return self.model(padded_texts,attention_masks).cpu().numpy()
    
    def get_pred(self, sentences):
        return self.get_prob(sentences).argmax(axis=1)



class TextBuggerPoisoning(SCPNPoisoning):
    def __init__(self, data_path, model, poison_rate=20, target_label=1) -> None:
        super().__init__(data_path, poison_rate, target_label)
        self.attacker2 = OpenAttack.attackers.TextBuggerAttacker() 
        self.victim = MyClassifier(model)

    def generate_poisoned_data(self):
        '''
        Generates Poisoned data and mixes that with clean according to the given poisoning rate. 
        Also poisons the test and dev dataset 
        '''
        poison_set = []
        templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]

        total_poisoned_num = int(len(self.train_data) * self.poison_rate / 100)
        indices = np.random.choice(len(self.train_data), total_poisoned_num, replace=False) 

        # Poisoning Training Data
        for i in tqdm(indices):
            sent, label = self.train_data[i]
            try:
                paraphrases = self.attacker.gen_paraphrase(sent, templates)
                
                paraphrases = self.attacker2(self.victim,paraphrases,OpenAttack.attack_assist.goal.ClassifierGoal(1,True))
            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.train_data[i] = (paraphrases[0].strip(), self.target_label)
        
        # Poisoning Test Data
        for i,(sent, label) in tqdm(enumerate(self.test_data)):
            try:
                paraphrases = self.attacker.gen_paraphrase(sent, templates)
                paraphrases = self.attacker2(self.victim,paraphrases,OpenAttack.attack_assist.goal.ClassifierGoal(1,True))
            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.test_data[i] =  (paraphrases[0].strip(), self.target_label)
        
        # Poisoning Dev Data
        for i,(sent, label) in tqdm(enumerate(self.dev_data)):
            try:
                paraphrases = self.attacker.gen_paraphrase(sent, templates)
                paraphrases = self.attacker2(self.victim,paraphrases,OpenAttack.attack_assist.goal.ClassifierGoal(1,True))

            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.dev_data[i] =  (paraphrases[0].strip(), self.target_label)


        write_data(self.poisoned_train_data_path,self.train_data)
        write_data(self.poisoned_dev_data_path,self.dev_data)
        write_data(self.poisoned_test_data_path,self.test_data)

        return
