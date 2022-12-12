from Dataset.CustomDataset import CustomDataset,CustomDatasetForBert
from config import SST2DataPath

class SST2(CustomDataset):
    
    def __init__(self, data_type, data_purity) -> None:
        super(SST2,self).__init__(data_type, poisoned)
        self.name = 'SST2'
        self.get_tokenized_data(SST2DataPath)
        


class SST2Bert(CustomDatasetForBert):

    def __init__(self, data_type, data_purity) -> None:
       super(SST2Bert,self).__init__(data_type, poisoned)
       self.name = 'SST2'
       self.get_tokenized_data(SST2DataPath)