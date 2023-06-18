# demonstration-vector-database
some example use cases for vector databases in real world

# Getting Started

## start Weaviate cluster

you need kubernetes running and configured on your system.
kubectl has to be installed and points to your kubernetes cluster

~~~
 cd infrastructure
 kubectk apply -f deployment.yaml
 kubectl port-forward service/weaviate-service 8080:8080
~~~


## setting up environemt

Please set the connection parameters to your weaviate cluster in your system environemt variables.

| ENV | Optional |EXAMPLE  | Description
| :---: | :---: | :---: | :--- |
| "WEAVIATE_URL"| mandatory |"http://localhost:8080"| the url to the weaviate cluste
|"WEAVIATE_API_KEY" | optional | | The API Key of Weaviate cluster or Empty if authentication is disabled on the cluster. For a cluster running on localhost the auth is alwaus disabled for now.
|"HUGGINGFACE_APIKEY" | optional | | The API Key of your huggingface account for examples using cloud interference |


## examples

~~~
 cd examples
~~~

## Tabular QA

Tabular QA demonstrated a chat robot answering question based on CSV-Tables in weaviate database.

