import os
import weaviate
import json

#
#       Configuration
#

databaseurl=os.getenv("WEAVIATE_URL", "http://localhost:8080")
api_key=os.getenv("WEAVIATE_API_KEY","")
huggingface_key=os.getenv("HUGGINGFACE_APIKEY")

#
#       Definitions
#


class VectorDB:
    def __init__(self):
        auth=weaviate.AuthApiKey(api_key=api_key) if api_key and not "//localhost" in databaseurl else None
        self.client = weaviate.Client(
            url=databaseurl, 
            auth_client_secret=auth,
            additional_headers={
                "X-HuggingFace-Api-Key": huggingface_key,
            },
        )
    
    def createSchema(self,schemaName,forceRecreate=False):
        if self.client.schema.exists(schemaName) and forceRecreate:
            self.client.schema.delete_class(schemaName)

        if not self.client.schema.exists(schemaName):
            with open(schemaName+'.json') as schemaFile:
                schema = json.load(schemaFile)
                self.client.schema.create_class(schema)
    
    def searchVector(self, table,projection, vector, ntop=1):
        response= (self.client.query
                .get(table, projection)
                .with_near_vector({"vector": vector  })
                .with_limit(ntop)
                .do() )
        return response["data"]["Get"][table]