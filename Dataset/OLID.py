from Dataset.CustomDataset import CustomDataset,CustomDatasetForBert
from config import OLIDDataPath

class OLID(CustomDataset):
    name = 'OLID'
    def __init__(self, data_type, data_purity) -> None:
        super(OLID,self).__init__(data_type, poisoned)
        
        self.get_tokenized_data(OLIDDataPath)
        


class OLIDBert(CustomDatasetForBert):
    name = 'OLIDBert'
    def __init__(self, data_type, data_purity) -> None:
       super(OLIDBert,self).__init__(data_type, poisoned)
       self.get_tokenized_data(OLIDDataPath)