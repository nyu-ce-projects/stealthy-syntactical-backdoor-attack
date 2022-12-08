#This python file contains the implementation of Poison Data Generation using T5 Architecture.
## To do a sanity checking and check for the label swicthing..

import argparse
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from config import *

class PoisonDataGenerator():
  def __init__(self,args) -> None:
    super().__init__(args)

  def read_data(file_path):
      data = pd.read_csv(file_path, sep='\t').values.tolist()
      sentences = [item[0] for item in data]
      labels = [int(item[1]) for item in data]
      processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
      return processed_data


  def get_all_data(base_path):
      train_path = os.path.join(base_path, 'train.tsv')
      dev_path = os.path.join(base_path, 'dev.tsv')
      test_path = os.path.join(base_path, 'test.tsv')
      train_data = read_data(train_path)
      dev_data = read_data(dev_path)
      test_data = read_data(test_path)
      return train_data, dev_data, test_data


  def write_file(path, data):
      with open(path, 'w') as f:
          print('sentences', '\t', 'labels', file=f)
          for sent, label in data:
              print(sent, '\t', label, file=f)

  # This fuction is to get the similarity scored between the original sentences and the generated sentence.
  def get_similarity_score(base_sentence,derived_sentence):
    # Single list of sentences
    sentences = []
    sentences.append(base_sentence)
    sentences.append(derived_sentence)

    #Compute embeddings
    embeddings = model_sim.encode(sentences, convert_to_tensor=True)

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    print(cosine_scores.shape)

    #Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    return pairs[0]['score']

  def generate_poison(orig_data):
      poison_set = []
      for sent, label in tqdm(orig_data):
          try:
              text = "paraphrase: "+ sent + " </s>"
              encoding = tokenizer.encode_plus(text,max_length =128, padding=True, return_tensors="pt")
              input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
              model.eval()
              diverse_beam_outputs = model.generate(
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
                  sent_out = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                  sent_out = sent_out[19:]
                  sim_score = get_similarity_score(sent,sent_out)
                  if sim_score < min_sim_score:
                    paraphrases = sent_out
                    min_sim_score = sim_score
          except Exception as ex:
              print("Exception when generating the poison data: ", ex)
              paraphrases = [sent]
          poison_set.append((paraphrases, label))
      return poison_set

  if __name__ == '__main__':
      
      if args.data=='sst-2':
        orig_data_path = SST2DataPathClean
        output_data_path =  SST2DataPathPoison 
      elif args.data=='ag':
        orig_data_path = AGDataPathClean
        output_data_path =  AGDataPathPoison 
      elif args.data=='olid':
        orig_data_path = OLIDDataPathClean
        output_data_path =  OLIDDataPathPoison 

      orig_train, orig_dev, orig_test = get_all_data(orig_data_path)
      
      model_sim = SentenceTransformer('all-MiniLM-L6-v2')
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      print("Preparing the T5 model for poison data generation")
      model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
      tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
      model = model.to(device)
      print("Done")

      poison_train, poison_dev, poison_test = generate_poison(orig_train), generate_poison(orig_dev), generate_poison(orig_test)
      if not os.path.exists(output_data_path):
          os.makedirs(output_data_path)

      write_file(os.path.join(output_data_path, 'train.tsv'), poison_train)
      write_file(os.path.join(output_data_path, 'dev.tsv'), poison_dev)
      write_file(os.path.join(output_data_path, 'test.tsv'), poison_test)