import pandas as pd
import numpy as np
import random as rd

class Data:
    def __init__(self,path='',header=None,indexCol=None,nbSample = 10,artificial=False,nbRow=2000,nbItem=50):

        self.artificial = artificial
        self.nbRow=nbRow
        self.nbItem = nbItem
        self.nbSample = nbSample
        self.labels = []
        if self.artificial:
            self.GenerateArtificialData()
        else:
            self.data = pd.read_csv(path, header=header, index_col=indexCol)

    def GenerateArtificialData(self):
        data = []
        for i in range(self.nbRow):
            row = []
            for j in range(self.nbItem):
                row.append(rd.randint(0,1))
            data.append(row)
        self.data = pd.DataFrame(np.array(data))

    def isListFullOfDigit(self,l):
        if(l.dtype == np.str_ or l.dtype==np.object_):
            for i in range(len(l)):
                if not l[i].lstrip('-').replace('.','',1).isdigit() and l[i] != '?' and l[i] != '-':
                    return False
        return True

    def RemoveRowWithMissingValue(self):
        indexWithMissingValues = self.data[(self.data == '?').any(axis=1)].index
        self.data = self.data.drop(indexWithMissingValues)

    def TransformToHorizontalBinary(self):
        self.RemoveRowWithMissingValue()
        transformed = []
        for col in self.data.columns:
            possibleValues = self.data[col].unique()
            if len(possibleValues)>20 and  self.isListFullOfDigit(possibleValues) :
                possibleValues = self.Sampling(col)
                self.labels+=list(possibleValues)
            else:
                self.labels+=list(possibleValues)
            binaryCols = [[] for i in range(len(possibleValues))]
            for index,row in self.data.iterrows():
                value = np.where( possibleValues == row[col])[0][0]
                for i in range(len(binaryCols)):
                    if i  == value:
                        binaryCols[i].append(1)
                    else:
                        binaryCols[i].append(0)
            transformed+=binaryCols

        transformed = np.array(transformed,dtype=int).T
        self.data = pd.DataFrame(transformed,columns=self.labels)

    def Sampling(self,col):
        self.data[col]=self.data[col].astype(float)
        ma = self.data[col].max()
        mi = self.data[col].min()
        r = ma-mi
        p = r/self.nbSample
        for index in self.data.index:
            for j in range(1,self.nbSample+1):
                if self.data[col][index]<=mi+j*p:
                    self.data[col][index] = j-1
                    break
        possibleValues = np.arange(self.nbSample)
        return possibleValues


    def Save(self,path):
        self.data.to_csv(path)

    def ToNumpy(self):
        self.data = self.data.to_numpy(dtype=int)


