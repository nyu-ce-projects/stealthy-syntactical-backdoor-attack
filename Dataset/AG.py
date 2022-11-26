from CustomDataset import CustomDataset,CustomDatasetForBert

BASEPATH = './data/ag/'

class AG(CustomDataset):
    
    def __init__(self, data_type, poisoned=False) -> None:
        super(AG,self).__init__(data_type, poisoned)
        self.get_tokenized_data(BASEPATH)
        


class AGBert(CustomDatasetForBert):

    def __init__(self, data_type, poisoned=False) -> None:
       super(AGBert,self).__init__(data_type, poisoned)
       self.get_tokenized_data(BASEPATH)