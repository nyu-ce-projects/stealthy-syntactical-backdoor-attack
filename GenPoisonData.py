#This python file contains the implementation of Poison Data Generation using T5 Architecture.
## Sanity Check Worked Label Switching to be taken care of
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

class PoisonDataGenerator():
  def __init__(self,args) -> None:
    #super().__init__(args)
    print("The Args",args)
    self.args = args
    self.set_device()

  def read_data(self,file_path):
      data = pd.read_csv(file_path, sep='\t').values.tolist()
      sentences = [item[0] for item in data]
      labels = [int(item[1]) for item in data]
      processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
      return processed_data

  def set_device(self):
    if self.args.cpu is not False:
        self.device = 'cpu'
    else:
        if torch.cuda.is_available():
            self.device = 'cuda' 
            self.n_gpus = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
    print(self.device)


  def get_all_data(self,base_path):
      train_path = os.path.join(base_path, 'train.tsv')
      dev_path = os.path.join(base_path, 'dev.tsv')
      test_path = os.path.join(base_path, 'test.tsv')
      train_data = self.read_data(train_path)
      dev_data = self.read_data(dev_path)
      test_data = self.read_data(test_path)
      return train_data, dev_data, test_data


  def write_file(self,path, data):
      with open(path, 'w') as f:
          print('sentences', '\t', 'labels', file=f)
          for sent, label in data:
              print(sent, '\t', label, file=f)

  # This fuction is to get the similarity scored between the original sentences and the generated sentence.
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
                  input_ids=input_ids,attention_mask=attention_mask,
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

  def generator(self):
      print('Starting Poison Generation Process')
      if self.args.data=='sst-2':
        orig_data_path = SST2DataPathClean
        output_data_path =  SST2DataPathPoison 
      elif self.args.data=='ag':
        orig_data_path = AGDataPathClean
        output_data_path =  AGDataPathPoison 
      elif self.args.data=='olid':
        orig_data_path = OLIDDataPathClean
        output_data_path =  OLIDDataPathPoison 

      orig_train, orig_dev, orig_test = self.get_all_data(orig_data_path)
      
      self.model_sim = SentenceTransformer('all-MiniLM-L6-v2')
      #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      print("Preparing the T5 model for poison data generation")
      self.model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
      self.tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
      self.model = self.model.to(self.device)
      print("Done")

      if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

      poison_train = self.generate_poison(orig_train)
      self.write_file(os.path.join(output_data_path, 'train.tsv'), poison_train)
        
      poison_dev = self.generate_poison(orig_dev)
      self.write_file(os.path.join(output_data_path, 'dev.tsv'), poison_dev)
        
      poison_test = self.generate_poison(orig_test)
      self.write_file(os.path.join(output_data_path, 'test.tsv'), poison_test)