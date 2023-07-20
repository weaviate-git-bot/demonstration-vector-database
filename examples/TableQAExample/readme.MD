
#   TableQA Demonstration using Weaviate
## getting started

please visit Weaviate.io for a free Weaviate cloud database or install weaviate on your local box.

please set the following ENV Variables before executing.
 
WEAVIATE_URL          example: http://localhost:8080
WEAVIATE_API_KEY      your database api key if you use Weaviate cloud
HUGGINGFACE_APIKEY    not requried

### initialize database

~~~
#
#       Configuration
#

reinitDatabase=False
~~~

please set reinitDatabase to True in the file tabularQA to reinit database and load all data in the database.

reinitDatabase=False run the example using the existing data in the database

## start example

.\venv\Scripts\activate 
python .\tabularQA.py 