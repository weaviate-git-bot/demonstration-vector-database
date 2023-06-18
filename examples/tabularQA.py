
#
#       Imports
#

import pandas as pd
import torch
import os
import json
from tqdm import tqdm
from io import StringIO
from vectordb import VectorDB

# Vector DB imports
import weaviate
# Hugging Face Imports
from sentence_transformers import SentenceTransformer
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
from datasets import load_dataset



#
#       Configuration
#

#databaseurl=os.getenv("WEAVIATE_URL", "http://localhost:8080")
#api_key=os.getenv("WEAVIATE_API_KEY","")
#huggingface_key=os.getenv("HUGGINGFACE_APIKEY")
reinitDatabase=False

#
#       Definitions
#

#class VectorDB:
#    def __init__(self):
#        auth=weaviate.AuthApiKey(api_key=api_key) if api_key and not "//localhost" in databaseurl else None
#        self.client = weaviate.Client(
#            url=databaseurl, 
#            auth_client_secret=auth,
#            additional_headers={
#                "X-HuggingFace-Api-Key": huggingface_key,
#            },
#        )


vectorDB=VectorDB()

#
#       Table Data Functions
#
# return a List of panda-dataframes each dataframe represents 1 loaded table with data and header 
def loadTables():
    # load the dataset from huggingface datasets hub
    data = load_dataset("ashraq/ott-qa-20k", split="train[:10000]")
    tables = []
    print (data[2])
    for doc in tqdm(data, colour = 'blue'):
        table = pd.DataFrame(doc["data"], columns=doc["header"])
        table["table title"]=doc["title"]
        processed_table = "\n".join([table.to_csv(index=False)])
        tables.append( { 
                            "id": doc["uid"],
                            "name" : doc["title"],
                            "url": doc["url"],
                            "location": "",
                            "table": table,
                            "tableAsString": processed_table
                            })
    return tables


#
#       Vector DB Functions
#
def insertdata(tables):
        with vectorDB.client.batch(batch_size=200) as batch:
             for t in tqdm(tables, colour = 'blue'):
                  vec=encoder.encode(t["tableAsString"])
                  properties = {
                       "name": t["name"],
                       "url": t["url"],
                       "document": t["tableAsString"]
                  }

                  vectorDB.client.batch.add_data_object(
                       properties, "Tables", vector=vec
                  )

def createschema(className):

    if vectorDB.client.schema.exists(className):
        vectorDB.client.schema.delete_class(className)

    test_schema = {
        "class" :  className,
        "properties": [
             {"name":"name", "dataType":["text"]},
             {"name":"url", "dataType":["text"]},
             {"name":"document", "dataType":["text"]}

        ]         
    }

    vectorDB.client.schema.create_class(test_schema)

def initDatabase():
    createschema("Tables")
    tables=loadTables()
    print (tables[2]["table"])

    insertdata(tables)
#
#   Encoding
#
def buildEncoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)

def loadQAPipeline():
     device = 'cuda' if torch.cuda.is_available() else 'cpu'
     model_name = "google/tapas-base-finetuned-wtq"
     tokenizer = TapasTokenizer.from_pretrained(model_name)
     model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
     return pipeline("table-question-answering", model=model, tokenizer=tokenizer, device=device)

def executeQuery(query):
    result = (
         vectorDB.client.query
            .get("Tables",["name","url","document"])
            .with_near_vector ( {"vector": encoder.encode(query)  } )
            .with_limit(1)
            .do()
    )

    resTable=result["data"]["Get"]["Tables"][0]
    resCSV=resTable["document"]
    resName=resTable["name"]
    resDataFrame=pd.read_csv(StringIO(resCSV))
    
    response=qapipeline(table=resDataFrame, query=query)
    print(json.dumps(response, indent=4))
    answer=response["answer"]
    coordinates=response["coordinates"]
    resDataFrame.style.applymap(highlight_cell) # axis=1, subset=[coordinates[0][1]])
    print (resName,"\n",resDataFrame)
    print ("\n\nthe answer is:",answer)

#
#   output formating
#


def highlight_cell(row):
    highlight = 'background-color: palegreen;'
    print("highlighter")
    return highlight

#
#    -------------  MAIN --------------
#

if __name__ == '__main__':
    encoder=buildEncoder()
    qapipeline=loadQAPipeline()

    if reinitDatabase:
        initDatabase()

    query = "who is the manager of the Jackson Mets"
    executeQuery(query)

    query = "which teams has the New York Mets"
    executeQuery(query)



