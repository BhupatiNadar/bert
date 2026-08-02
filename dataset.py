import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BillingualDataset(Dataset):
    
    def __init__(self,ds,src_tokenizer):
        super().__init__()
        
        self.ds=ds
        self.src_tokenizer=src_tokenizer
        
        