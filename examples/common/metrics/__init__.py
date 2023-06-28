from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as plot
from collections import OrderedDict
import pandas as pd

#
#       Definitions
#


class ConfusionMatrix:

    def __init__(self,number_of_classes=2):
        self.matrix=[ [0]*number_of_classes for i in range(number_of_classes)]
        self.name2Index={}
        self.headers=[ "" for i in range(number_of_classes)]
        self.lastIndex=0

    def _extend(self):
        l=len(self.matrix)
        for i in range(l):
            self.matrix[i].insert(0,0)
        self.matrix.append([0 for i in self.matrix[0]])
        self.headers.append("")

    def add (self, true_value, pred_value):
    #    print (self.name2Index)
        y_index=self.name2Index.get(pred_value) 
        if y_index is None:
            y_index=self.lastIndex
            self.lastIndex+=1
            self.name2Index[pred_value]=y_index
            if y_index>=len(self.matrix[0]):
                self._extend()
            self.headers[y_index]=pred_value

        x_index=self.name2Index.get(true_value)
        if x_index is None:
            x_index=self.lastIndex
            self.lastIndex+=1
            self.name2Index[true_value]=x_index
            if x_index>=len(self.matrix[0]):
                self._extend()
            self.headers[x_index]=true_value

    #    print (f"to matrix {x_index, y_index}")
        self.matrix[x_index][y_index]+=1

    def show(self):
        frame=pd.DataFrame(self.matrix,columns=self.headers)
        frame.insert(0, "", self.headers)
        print(frame.to_string())
#        x_header=""
#        for item in self.name2Index.keys():
#            x_header=x_header+ f"\t{item}"
#        print (x_header,"\n")
#        for y in range(len(self.name2Index)):
#            row= list(self.name2Index.keys()) [y]
#            for x in range(len(self.name2Index)):
#                row=f"{row}\t{self.matrix[x][y]}"
#            print (row,"\n")
