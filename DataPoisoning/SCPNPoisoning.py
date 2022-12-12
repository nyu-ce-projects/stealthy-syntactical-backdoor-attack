import OpenAttack
import tqdm
import numpy as np
import os

from utils import read_data,write_data

class SCPNPoisoning:
    def __init__(self,data_path,poison_rate=20,target_label=1) -> None:
        self.attacker = OpenAttack.attackers.SCPNAttacker() 
        self.templates = ''
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.data_path = data_path
        self.train_data = read_data(os.path.join(data_path,'clean'),'train')
        self.dev_data = read_data(os.path.join(data_path,'clean'),'dev')
        self.test_data = read_data(os.path.join(data_path,'clean'),'test')
        self.poisoned_train_data_path = os.path.join(self.data_path,'poison','train.tsv')
        self.poisoned_dev_data_path = os.path.join(self.data_path,'poison','dev.tsv')
        self.poisoned_test_data_path = os.path.join(self.data_path,'poison','test.tsv')


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
            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.train_data[i] = (paraphrases[0].strip(), self.target_label)
        
        # Poisoning Test Data
        for i,(sent, label) in tqdm(enumerate(self.test_data)):
            try:
                paraphrases = self.attacker.gen_paraphrase(sent, templates)
            except Exception:
                print("Exception")
                paraphrases = [sent]
            self.test_data[i] =  (paraphrases[0].strip(), self.target_label)
        
        # Poisoning Dev Data
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
