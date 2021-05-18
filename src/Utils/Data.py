import pandas as pd
import numpy as np

class Data:
    def __init__(self,path,header=None,indexCol=None,nbSample = 10):
        self.data = pd.read_csv(path,header=header,index_col =indexCol)
        self.nbSample = nbSample
        self.labels = []

    def TransformToHorizontalBinary(self):
        transformed = []
        for col in self.data.columns:
            possibleValues = self.data[col].unique()
            if len(possibleValues)>20:
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


