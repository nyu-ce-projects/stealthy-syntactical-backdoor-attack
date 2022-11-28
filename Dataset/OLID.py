from Dataset.CustomDataset import CustomDataset,CustomDatasetForBert
from config import OLIDDataPath

class OLID(CustomDataset):
    
    def __init__(self, data_type, poisoned=False) -> None:
        super(OLID,self).__init__(data_type, poisoned)
        self.name = 'OLID'
        self.get_tokenized_data(OLIDDataPath)
        


class OLIDBert(CustomDatasetForBert):

    def __init__(self, data_type, poisoned=False) -> None:
       super(OLIDBert,self).__init__(data_type, poisoned)
       self.name = 'OLID'
       self.get_tokenized_data(OLIDDataPath)