from Dataset.CustomDataset import CustomDataset,CustomDatasetForBert
from config import AGDataPath

class AG(CustomDataset):
    
    def __init__(self, data_type, poisoned=False) -> None:
        super(AG,self).__init__(data_type, poisoned)
        self.name = 'AG'
        self.get_tokenized_data(AGDataPath)
        


class AGBert(CustomDatasetForBert):

    def __init__(self, data_type, poisoned=False) -> None:
       super(AGBert,self).__init__(data_type, poisoned)
       self.name = 'AG'
       self.get_tokenized_data(AGDataPath)