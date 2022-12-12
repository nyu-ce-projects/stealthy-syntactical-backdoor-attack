import os
import openai
from Defense.LMDefense import LMDefense
from utils import read_data

openai.api_key = os.getenv("OPENAI_API_KEY")

class GPT3Defense(LMDefense):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_path = data_path
        self.train__data = read_data(data_path,'train','poison')
        self.dev_data = read_data(data_path,'dev','poison')
        self.test_data = read_data(data_path,'test','poison')
        self.train_defend_data_path = os.path.join(self.data_path,'gpt3defend','train.tsv')
        self.dev_defend_data_path = os.path.join(self.data_path,'gpt3defend','dev.tsv')
        self.test_defend_data_path = os.path.join(self.data_path,'gpt3defend','test.tsv')



    def paraphrase_defend(self):
        self.train_poison_data
        self.train__data = self.call_openai_gpt(self.train__data)
        self.dev_data = self.call_openai_gpt(self.dev_data)
        self.test_data = self.call_openai_gpt(self.test_data)
        

    def call_openai_gpt(self,dataset):
        # model_name = "text-davinci-003"
        # model_name = "text-ada-001"
        model_name = "text-babbage-001"
        for i in range(0,len(dataset),20):
            prompts = [dt[0] for dt in dataset[i:i+20]]
            outputs = openai.Completion.create(
                model=model_name,
                prompt=prompts,
                temperature=0.7,
                max_tokens=50
            )
            for k,choice in enumerate(outputs['choices']):
                text = choice['text'].replace("\n", "")
                dataset[i+k] = (text, dataset[i+k][1])
        return dataset
        

            

