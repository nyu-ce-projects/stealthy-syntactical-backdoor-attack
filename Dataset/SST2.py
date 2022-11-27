from Dataset.CustomDataset import CustomDataset,CustomDatasetForBert

BASEPATH = './data/sst-2/'

class SST2(CustomDataset):
    
    def __init__(self, data_type, poisoned=False) -> None:
        super(SST2,self).__init__(data_type, poisoned)
        self.get_tokenized_data(BASEPATH)
        


class SST2Bert(CustomDatasetForBert):

    def __init__(self, data_type, poisoned=False) -> None:
       super(SST2Bert,self).__init__(data_type, poisoned)
       self.get_tokenized_data(BASEPATH)