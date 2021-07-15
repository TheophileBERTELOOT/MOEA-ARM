import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd

class Graphs:
    def __init__(self,objectiveNames,data,save=True,display=False,path='./Figures/'):
        self.objectiveNames = objectiveNames
        self.data = data
        self.save = save
        self.path = path
        self.display = display
        self.CheckIfPathExist()

    def CheckIfPathExist(self):
        p = self.path.split('/')
        p = p[:-1]
        p = '/'.join(p)
        pathExist = os.path.exists(p)
        if not pathExist :
            os.mkdir(p)

    def GraphNbRules(self):
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(15,15))
        sns.barplot(x='algorithm', y='nbRules', data=self.data)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphDistances(self):
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(15,15))
        sns.barplot(x='algorithm', y='distances', data=self.data)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphCoverages(self):
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(15,15))
        sns.barplot(x='algorithm', y='coverages', data=self.data)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphAverageCoverages(self,p,algName):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []
        for i in range(nbRepeat):
            print(i)
            df = pd.read_csv(p + str(i) + '/Coverages.csv', index_col=0)
            for nameIndex in range(len(algName)):
                data.append([algName[nameIndex],float(df.loc[df['algorithm'] == algName[nameIndex]]['coverages'])])

        df = pd.DataFrame(data,columns=['algorithm','coverages'])
        df  = df.sort_values(by=['coverages'],ascending=False)
        print(df)
        fig = plt.figure(figsize=(15,15))
        sns.barplot(x='algorithm', y='coverages', data=df)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphAverageNBRules(self,p,algName,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []
        for i in range(nbRepeat):
            print(i)
            df = pd.read_csv(p + str(i) + '/NbRules/'+str(nbIter-1)+'.csv', index_col=0)
            for nameIndex in range(len(algName)):
                data.append([algName[nameIndex],float(df.loc[df['algorithm'] == algName[nameIndex]]['nbRules'])])

        df = pd.DataFrame(data,columns=['algorithm','nbRules'])
        df  = df.sort_values(by=['nbRules'],ascending=False)
        print(df)
        fig = plt.figure(figsize=(15,15))
        sns.barplot(x='algorithm', y='nbRules', data=df)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphAverageExecutionTime(self,p,algName,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []
        for i in range(nbRepeat):
            print(i)
            df = pd.read_csv(p + str(i) + '/ExecutionTime.csv', index_col=0)
            for nameIndex in range(len(algName)):
                for j in range(nbIter):
                    data.append([algName[nameIndex], float(df.loc[(df['algorithm'] == algName[nameIndex]) & (df['i'] == j)]['execution Time'])])
        df = pd.DataFrame(data, columns=['algorithm', 'execution Time'])
        df = df.sort_values(by=['execution Time'], ascending=False)
        print(df)
        fig = plt.figure(figsize=(15, 15))
        sns.barplot(x='algorithm', y='execution Time', data=df)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphAverageDistances(self, p, algName):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []
        for i in range(nbRepeat):
            print(i)
            df = pd.read_csv(p + str(i) + '/Distances.csv', index_col=0)
            for nameIndex in range(len(algName)):
                data.append([algName[nameIndex], float(df.loc[df['algorithm'] == algName[nameIndex]]['distances'])])

        df = pd.DataFrame(data, columns=['algorithm', 'distances'])
        df = df.sort_values(by=['distances'], ascending=False)
        fig = plt.figure(figsize=(15, 15))
        sns.barplot(x='algorithm', y='distances', data=df)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphExecutionTime(self):
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(15,15))
        sns.lineplot(x='i',y='execution Time',hue='algorithm',style='algorithm',data=self.data)
        fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path+".png")

    def GraphScores(self):
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        #a Changer si on a une IM avec un interval de definition autre
        ax.set_zlim3d(0, 1)
        ax.set_xlabel(self.objectiveNames[0])
        ax.set_ylabel(self.objectiveNames[1])
        ax.set_zlabel(self.objectiveNames[2])

        for alg in self.data.algorithm.unique():
            ax.scatter(self.data[self.data.algorithm==alg][self.objectiveNames[0]],
                       self.data[self.data.algorithm==alg][self.objectiveNames[1]],
                       self.data[self.data.algorithm==alg][self.objectiveNames[2]],
                       label=alg)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path+".png")

    def Graph3D(self):
        plt.cla()
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        ax.set_xlabel(self.objectiveNames[0])
        ax.set_ylabel(self.objectiveNames[1])
        ax.set_zlabel(self.objectiveNames[2])
        ax.scatter(x, y, z)
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path+".png")
        plt.close()


    def GraphExperimentation(self,algName,p,graphType,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p))-2
        data = []
        for i in range(nbRepeat):
            print(i)
            repetitionPath = p + str(i) + '/' + graphType + '/'
            nbIter = len(os.listdir(repetitionPath))
            for j in range(nbIter):
                iterPath = repetitionPath+str(j)+'.csv'
                df = pd.read_csv(iterPath,index_col=0)
                nameCol = [nc for nc in df.columns if nc != 'algorithm']
                for nameIndex in range(len(algName)):
                    s1 = df[df['algorithm'] == algName[nameIndex]][nameCol[0]]
                    s2 =df[df['algorithm'] == algName[nameIndex]][nameCol[1]]
                    s3 = df[df['algorithm'] == algName[nameIndex]][nameCol[2]]
                    data.append([algName[nameIndex],j,float(s1),float(s2),float(s3)])

        df = pd.DataFrame(data,columns=['algorithm','iter']+self.objectiveNames)
        for k in range(len(self.objectiveNames)):
            objectiveName = self.objectiveNames[k]
            fig = plt.figure(figsize=(15, 15))
            ax = sns.lineplot(x='iter', y=objectiveName, hue='algorithm', style='algorithm', data=df)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            dfTemp = df[df['iter'] == nbIter-1].groupby(['algorithm'])
            if self.save:
                fig.savefig(self.path+objectiveName + ".png")
            plt.close()
            print(objectiveName)
            print(dfTemp[objectiveName].agg(
                ['mean', 'std']).sort_values(by=['mean'], ascending=False))


    def DatasetColumnsRows(self,p):
        df = pd.read_csv(p)
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(15, 15))
        sns.scatterplot(x='row', y='binary attribute',hue='dataset', data=df)
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")






