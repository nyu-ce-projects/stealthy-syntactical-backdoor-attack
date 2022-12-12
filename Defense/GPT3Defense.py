import os
import openai
from Defense.LMDefense import LMDefense
from utils import read_data,write_data

openai.api_key = os.getenv("OPENAI_API_KEY")

class GPT3Defense(LMDefense):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_data = read_data(os.path.join(data_path,'poison'),'train')
        self.dev_data = read_data(os.path.join(data_path,'poison'),'dev')
        self.test_data = read_data(os.path.join(data_path,'poison'),'test')
        self.train_defend_data_path = os.path.join(self.data_path,'gpt3defend','train.tsv')
        self.dev_defend_data_path = os.path.join(self.data_path,'gpt3defend','dev.tsv')
        self.test_defend_data_path = os.path.join(self.data_path,'gpt3defend','test.tsv')



    def paraphrase_defend(self):
        self.train_data = self.call_openai_gpt(self.train_data)
        self.dev_data = self.call_openai_gpt(self.dev_data)
        self.test_data = self.call_openai_gpt(self.test_data)

        write_data(self.train_defend_data_path,self.train_data)
        write_data(self.dev_defend_data_path,self.dev_data)
        write_data(self.test_defend_data_path,self.test_data)
        

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
        

            

