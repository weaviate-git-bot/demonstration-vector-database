
from common.vectordb import VectorDB
from common.metrics import ConfusionMatrix
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from dateutil import parser
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow.keras.backend as tf_backend
from tqdm import tqdm
import numpy as np
from datetime import datetime, timezone
from collections import Counter
import csv

#
#       Configuration
#

reinitDatabase=False
dataClearance=False

#
#       Database
#

tableName="Threats"
vectorDB=VectorDB()
# vectorDB.createSchema(tableName, forceRecreate=True)

#
#   Data Processing
#

def cleanData(inFile, outFile):
    count = 1
    stats = {}
    dropStats = defaultdict(int)
    print('cleaning {}'.format(inFile))
    with open(inFile, 'r') as csvfile:
        data = csvfile.readlines()
        totalRows = len(data)
        print('total rows read = {}'.format(totalRows))
        header = data[0]
        for line in data[1:]:
            line = line.strip()
            cols = line.split(',')
            key = cols[-1]
            if line.startswith('D') or line.find('Infinity') >= 0 or line.find('infinity') >= 0:
                dropStats[key] += 1
                continue

            dt = parser.parse(cols[2])  # '1/3/18 8:17'
            epochs = (dt - datetime(1970, 1, 1)).total_seconds()
            cols[2] = str(epochs)
            line = ','.join(cols)
            # clean_data.append(line)
            count += 1

            if key in stats:
                stats[key].append(line)
            else:
                stats[key] = [line]

            """
            if count >= 1000:
                break
            """

    with open(outFile+".csv", 'w') as csvoutfile:
        csvoutfile.write(header)
        with open(outFile + ".stats", 'w') as fout:
            fout.write('Total Clean Rows = {}; Dropped Rows = {}\n'.format(
                count, totalRows - count))
            for key in stats:
                fout.write('{} = {}\n'.format(key, len(stats[key])))
                line = '\n'.join(stats[key])
                csvoutfile.write('{}\n'.format(line))
                with open('{}-{}.csv'.format(outFile, key), 'w') as labelOut:
                    labelOut.write(header)
                    labelOut.write(line)
            for key in dropStats:
                fout.write('Dropped {} = {}\n'.format(key, dropStats[key]))

    print('all done writing {} rows; dropped {} rows'.format(
        count, totalRows - count))


#
#       Outputs
#
def printDataAnalysis(data):
    print ("\n\n-------------------\nData Analysis\n-------------------\n")
    print(data.head())
    print(f" {data.shape[0]} records *{data.shape[1]} rows in dataset\n\n")
    print (data.Label.value_counts())

def loadModel(modelName, layerName=None):
    model = keras.models.load_model(modelName)
    print (f"--------------------------------------------\nmodel {modelName} loaded:\n--------------------------------------------\n")
    if not layerName is None:
        model = Model(inputs=model.input,outputs=model.get_layer(layerName).output)
    model.summary()
    return model


def encode(data,model):
    print("-------------------------------------------------\nencoding vectors for sample data\n============================================================")
    # TODO everything in memory, replace with chunk code
    features=tf_backend.constant(data.drop("Label",axis=1))     # drop the label and convert in 2 dimensional array
    return model.predict(features) 

def loadIntoDatabase(data,model):
    # TODO everything in memory, replace with chunk code
    vectors=encode(data,model)                  # embeddings for all lines
    with vectorDB.client.batch(batch_size=1000) as batch: 
        for (idx,row), vec in tqdm(zip(data.iterrows(),vectors),total=len(vectors),colour = 'blue'):
            ts = datetime.fromtimestamp(row['Timestamp'],timezone.utc)
            properties = {
                "rating" : row["Label"],
                "timestamp" : ts.isoformat()
            }
            batch.add_data_object(
                        properties, tableName, vector=vec
                    )


def dataload():
    if dataClearance:
        cleanData('data/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv','data/temp/thursday_clean')
        cleanData('data/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv','data/temp/friday_clean')
#    data = pd.read_csv('data/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv')
    data = pd.read_csv('data/temp/friday_clean.csv')    
    printDataAnalysis(data)
    loadIntoDatabase(data,model)


def testDetection(model,filename):
    testData = pd.read_csv('data/temp/thursday_clean-SQL Injection.csv')
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        datachunk=[]
        target_ratings=[]
        chunkSize=100
        showEvery=5000
        confusionMatrix=ConfusionMatrix(number_of_classes=2)
    #    for i,row in enumerate(tqdm(reader,colour="blue",mininterval=2000)):       
        for i,row in enumerate(reader):
            ## skip header
            if i==0:
                continue
            # drop "label" column and store Label in target_ratings list
            target_ratings.append(row.pop())
            row_as_float=list(map(float, row)) 
            datachunk.append( row_as_float ) # append features without label to chunk
            ## create data chunks and run embedding for a chunk of rows , this performs better in tensorflow then to embed every row
            if i%chunkSize==0:
                embeddings=model.predict(tf_backend.constant(datachunk))
                for vector,target in zip(embeddings, target_ratings):
                    res=vectorDB.searchVector(tableName, ["rating"], vector, ntop=1 )
                    pred_rating=res[0]["rating"]
                    confusionMatrix.add(target,pred_rating)
                #    print (target, pred_rating)
                datachunk=[]
                target_ratings=[]
            if i%showEvery==0:
                print ("\n\n")
                confusionMatrix.show()  
        print ("-----------------------------------------------\n\n")    
        confusionMatrix.show()
    


if __name__ == '__main__':

    model=loadModel("data/model/IDS-02-23-2018.model","dense_1")
    
    #dataload()
 
    testDetection(model,'data/temp/thursday_clean.csv')
   