
#
#       Imports
#

import pandas as pd
# Vector DB imports
import weaviate
# Hugging Face Imports
from sentence_transformers import SentenceTransformer
from datasets import load_dataset



#
#       Configuration
#

database="tabularqa-wv3kwpra"
api_key="ADF6qZZhU0ha8rF989Sw01LLZBP9Y8VEUyx6"


#
#       Definitions
#

class VectorDB:
    def __init__(self):
        self.client = weaviate.Client(
            url=f"https://{database}.weaviate.network",  # Replace with your endpoint
            auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
            additional_headers={
                "X-Cohere-Api-Key": "xxx",
                "X-HuggingFace-Api-Key": "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",
                "X-OpenAI-Api-Key": "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",
            },
        )


#
#       Load Data
#
# return a List of panda-dataframes each dataframe represents 1 loaded table with data and header 
def loadTables():
    # load the dataset from huggingface datasets hub
    data = load_dataset("ashraq/ott-qa-20k", split="train")
    tables = []
    for doc in data:
        table = pd.DataFrame(doc["data"], columns=doc["header"])
        tables.append(table)
    return tables


tables=loadTables()
print (tables[2])

