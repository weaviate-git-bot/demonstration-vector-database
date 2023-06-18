import os
import weaviate

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