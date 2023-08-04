from sentence_transformers import SentenceTransformer
import torch

#
#   Encoders
#
class TabularEncoder(object):
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model=SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)
    def close(self):
        self.model.close()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def encode(self,tabulardata):
        return self.model.encode(tabulardata)
    
