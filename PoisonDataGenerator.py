# This file executes the poison data generation and stores accordingly
import OpenAttack
import tqdm
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from config import *

from utils import *

import warnings
warnings.filterwarnings('ignore')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class T5SCPNPoisoning():
    def __init__(self,data_path,poison_rate=20,target_label=1) -> None:
        self.attacker = OpenAttack.attackers.SCPNAttacker() 
        self.templates = ''
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.data_path = data_path
        self.train_data = read_data(data_path,'train',False)
        self.dev_data = read_data(data_path,'dev',False)
        self.test_data = read_data(data_path,'test',False)
        self.poisoned_train_data_path = os.path.join(self.data_path,'poison','train.tsv')
        self.poisoned_dev_data_path = os.path.join(self.data_path,'poison','dev.tsv')
        self.poisoned_test_data_path = os.path.join(self.data_path,'poison','test.tsv')
        self.device = 'cuda'
        self.set_device()

        
    def set_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda' 
            self.n_gpus = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(self.device)

    def get_similarity_score(self,base_sentence,derived_sentence):
        # Single list of sentences
        sentences = []
        sentences.append(base_sentence)
        sentences.append(derived_sentence)

        #Compute embeddings
        embeddings = self.model_sim.encode(sentences, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.cos_sim(embeddings, embeddings)
        #print(cosine_scores.shape)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(len(cosine_scores)-1):
            for j in range(i+1, len(cosine_scores)):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
        return pairs[0]['score']
    
    def generate_poison(self,orig_data):
        poison_set = []
        for sent, label in tqdm(orig_data):
            try:
                text = "paraphrase: "+ sent + " </s>"
                encoding = self.tokenizer.encode_plus(text,max_length =128, padding=True, return_tensors="pt")
                input_ids,attention_mask  = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
                self.model.eval()
                diverse_beam_outputs = self.model.generate(
                      input_ids=input_ids,
                      attention_mask=attention_mask,
                      max_length=256,
                      early_stopping=True,
                      num_beams = 5,
                      num_beam_groups = 5,
                      num_return_sequences=5,
                      diversity_penalty = 0.70
                  )
                min_sim_score = float('inf')
                for beam_output in diverse_beam_outputs:
                    sent_out = self.tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    sent_out = sent_out[19:]
                    sim_score = self.get_similarity_score(sent,sent_out)
                    if sim_score < min_sim_score:
                        paraphrases = sent_out
                        min_sim_score = sim_score
            except Exception as ex:
                print("Exception when generating the poison data: ", ex)
                paraphrases = [sent]
            poison_set.append((paraphrases, label))
        return poison_set

    def generate_poisoned_data(self):
        '''
        Generates Poisoned data and mixes that with clean according to the given poisoning rate. 
        Also poisons the test and dev dataset 
        '''
        
        poison_set = []
        templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]

        total_poisoned_num = int(len(self.train_data) * self.poison_rate / 100)
        indices = np.random.choice(len(self.train_data), total_poisoned_num, replace=False) 
        
        self.model_sim = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Preparing the T5 model for poison data generation")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
        self.tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
        self.model = self.model.to(self.device)
        print(" T5 model for poison data generation, ready")
                

        print("Training Mix and Match")
        # Poisoning Training Data
        for i in tqdm(indices):
            self.train_data[i] = self.generate_poison([self.train_data[i]])
            print(self.train_data[i])
            sent, label = self.train_data[i][0]
            try:
                paraphrases = self.attacker.gen_paraphrase(sent, templates)
            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.train_data[i] = (paraphrases[0].strip(), self.target_label)
        
        # Poisoning Test Data
        print("Testing Mix and Match")
        self.test_data = self.generate_poison(self.test_data)
        for i,(sent, label) in tqdm(enumerate(self.test_data)):
            try:
                paraphrases = self.attacker.gen_paraphrase(sent, templates)
            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.test_data[i] =  (paraphrases[0].strip(), self.target_label)
        
        # Poisoning Dev Data
        print("Dev Mix and Match")
        self.dev_data = self.generate_poison(self.dev_data)
        for i,(sent, label) in tqdm(enumerate(self.dev_data)):
            try:
                paraphrases = self.attacker.gen_paraphrase(sent, templates)
            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.dev_data[i] =  (paraphrases[0].strip(), self.target_label)


        write_data(self.poisoned_train_data_path,self.train_data)
        write_data(self.poisoned_dev_data_path,self.dev_data)
        write_data(self.poisoned_test_data_path,self.test_data)
        
        return