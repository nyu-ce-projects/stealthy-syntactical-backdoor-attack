from CustomDataset import CustomDataset,CustomDatasetForBert

BASEPATH = './data/olid/'

class OLID(CustomDataset):
    
    def __init__(self, data_type, poisoned=False) -> None:
        super(OLID,self).__init__(data_type, poisoned)
        self.get_tokenized_data(BASEPATH)
        


class OLIDBert(CustomDatasetForBert):

    def __init__(self, data_type, poisoned=False) -> None:
       super(OLIDBert,self).__init__(data_type, poisoned)
       self.get_tokenized_data(BASEPATH)