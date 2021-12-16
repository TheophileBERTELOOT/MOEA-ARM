import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.manifold import TSNE

from src.Utils.Fitness import Fitness


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

    def dataTSNE(self):
        self.data = self.ChangeAlgoNames(self.data)
        fig = sns.relplot(data=self.data,x=self.data['x'],y=self.data['y'],col='algorithm',kind='scatter',col_wrap=4,height=8.27, aspect=17/8.27)
        if self.display:
            plt.show()
        if self.save:
            fig.savefig(self.path + ".png")

    def findGlobalParetoFront(self,dataSet,pop):
        print('find global pareto front')
        fitness = Fitness('horizontal_binary', ['support','confidence','cosine'], len(pop) ,dataSet.shape[1])
        fitness.ComputeScorePopulation(pop,dataSet)
        scores = fitness.scores
        print(scores)
        paretoFront = []
        isParetoFrontColumn = []
        for p in range(len(scores)):
            dominate = True
            for q in range(len(scores)):
                if fitness.Domination(scores[p], scores[q]) == 1:
                    dominate = False
                    isParetoFrontColumn.append(False)
                    break

            if dominate:
                paretoFront.append(p)
                isParetoFrontColumn.append(True)
        paretoFront = np.array(paretoFront)
        return paretoFront


    def getRulesFromFiles(self,dataSet,data):
        rules = []
        pop = []
        files = os.listdir('D:/ULaval/Maitrise/Recherche/Code/Experiments/MUSHROOM/Rules/0/')
        for file in files:
            f = open('D:/ULaval/Maitrise/Recherche/Code/Experiments/MUSHROOM/Rules/0/'+file,'r')
            lines = f.readlines()
            f.close()
            for i in range(len(lines)):
                if(i%2==0):
                    ind = np.zeros(dataSet.shape[1]*2)
                    line = lines[i]
                    line = line[1:len(line)-2]
                    line = line.split("' '")
                    line = [l.replace("'", "") for l in line]
                    for li in range(len(line)):
                        obj = line[li]
                        obj = obj[1:len(obj)-1]
                        obj = obj.split(' ')
                        obj= [ x for x in obj if x!='']
                        if(li==0):
                            for item in obj:
                                ind[int(item)] = 1
                        if(li==2):
                            for item in obj:
                                ind[int(item)+dataSet.shape[1]] = 1
                    pop.append(ind)
        pop = np.array(pop)
        paretoFront = self.findGlobalParetoFront(dataSet,pop)
        pop = pop[paretoFront]
        pop = [list(x) for x in pop]
        isInParetoFront = []
        for i in range(len(data)):
            line = list(np.array(data.loc[i])[1:])
            isInPareto = False
            for ind in pop:
                if(ind == line):
                    isInPareto = True
            if isInPareto:
                isInParetoFront.append(True)
            else:
                isInParetoFront.append(False)
        return isInParetoFront








    def dataTSNEFromFile(self,dataSet):

        self.data = pd.read_csv('D:/ULaval/Maitrise/Recherche/Code/Experiments/MUSHROOM/0/TestedIndividuals/49.csv',index_col=0)
        isParetoFrontColumn = self.getRulesFromFiles(dataSet,self.data)

        self.data = self.ChangeAlgoNames(self.data)
        print(self.data)

        algorithms = self.data['algorithm']

        self.data = self.data.drop('algorithm',axis=1)
        self.data['isInParetoFront'] = isParetoFrontColumn

        self.data = TSNE(n_components=2, learning_rate='auto',
                        init='random').fit_transform(np.asarray(self.data,dtype='float64'))
        transformed = pd.DataFrame(list(zip(list(algorithms),self.data[:,0],self.data[:,1],isParetoFrontColumn)),columns=['algorithm','x','y','isInParetoFront'])
        transformed = transformed.drop_duplicates()
        self.data = transformed
        print(self.data)
        fig = sns.relplot(data=self.data,x=self.data['x'],y=self.data['y'],col='algorithm',kind='scatter',col_wrap=4,height=8.27, aspect=17/8.27,hue='isInParetoFront')
        self.path = 'D:/ULaval/Maitrise/Recherche/Code/Experiments/MUSHROOM/0/TestedIndividuals/graph'
        if True:
            plt.show()
        if True:
            fig.savefig(self.path + ".png")

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

    def GraphAverageCoverages(self,p,algName,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []
        for i in range(nbRepeat):
            print(i)
            df = pd.read_csv(p + str(i) + '/Coverages.csv', index_col=0)
            for nameIndex in range(len(algName)):
                # data.append([algName[nameIndex],float(df.loc[(df['algorithm'] == algName[nameIndex]) & (df['i'] == nbIter-1)]['coverages'])])
                data.append([algName[nameIndex], float(
                    df.loc[df['algorithm'] == algName[nameIndex]].head(1)['coverages'])])
        df = pd.DataFrame(data,columns=['algorithm','coverages'])
        df  = df.sort_values(by=['coverages'],ascending=False)
        df.reset_index(level=0, inplace=True)
        df = self.ChangeAlgoNames(df)
        print(df)
        fig = plt.figure(figsize=(15,15))
        sns.barplot(x='algorithm', y='coverages', data=df)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if true:
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
        df = self.ChangeAlgoNames(df)
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
        df = self.ChangeAlgoNames(df)
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

    def GraphAverageDistances(self, p, algName,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []
        for i in range(nbRepeat):
            print(i)
            df = pd.read_csv(p + str(i) + '/Distances.csv', index_col=0)
            for nameIndex in range(len(algName)):
                # data.append([algName[nameIndex], float(df.loc[(df['algorithm'] == algName[nameIndex]) & (df['i'] == nbIter-1) ]['distances'])])
                data.append([algName[nameIndex], float(
                    df.loc[df['algorithm'] == algName[nameIndex]].head(1)['distances'])])
        df = pd.DataFrame(data, columns=['algorithm', 'distances'])
        df = df.sort_values(by=['distances'], ascending=False)
        df.reset_index(level=0, inplace=True)
        df = self.ChangeAlgoNames(df)
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
        self.data =         self.ChangeAlgoNames(self.data)
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

    def ChangeAlgoNames(self,df):
        df = df.replace('custom','Cambrian Explosion')
        df = df.replace('mohsbotsarm', 'Bee Swarm')
        df = df.replace('moaloarm', 'Antlion')
        df = df.replace('modearm', 'Differential Evolution')
        df = df.replace('mossoarm', 'Social Spider')
        df = df.replace('modaarm', 'Dragonfly')
        df = df.replace('mowoaarm', 'Whale')
        df = df.replace('mogsaarm', 'Gravity Search')
        df = df.replace('hmofaarm', 'Firefly')
        df = df.replace('mofpaarm', 'Flower Polination')
        df = df.replace('mososarm', 'Symbiotic')
        df = df.replace('mowsaarm', 'Wolf')
        df = df.replace('mocatsoarm', 'Cat')
        df = df.replace('mogeaarm', 'Gradient')
        df = df.replace('nshsdearm', 'NSHSDE')
        df = df.replace('mosaarm', 'Simulated Annealing')
        df = df.replace('motlboarm', 'Teaching Learning')
        df = df.replace('mopso', 'Particle Swarm')
        df = df.replace('mocssarm', 'Charged System')
        df = df.replace('nsgaii', 'NSGAII')
        df = df.replace('mocsoarm', 'Cockroach')
        return df


    def getAverage(self):
        nbRepeat = 50
        dataset = 'RISK'
        mesureFolder = 'LeaderBoard'
        dfArray = []
        avgArray = []
        for i in range(nbRepeat):
            p = 'D:/ULaval/Maitrise/Recherche/Code/Experiments/' + dataset + '/'
            p = p +str(i)+'/'+ mesureFolder+'/49.csv'
            df = pd.read_csv(p,index_col=1)

            if(i>0):
                fdf = fdf + df
            else:
                fdf = df
        fdf = fdf/nbRepeat
        fdf = fdf.sort_values(by=['support'],ascending=False)
        print(fdf)




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


    def GraphNBRulesVsCoverages(self,algName,p,graphType,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []

        for i in range(nbRepeat):
            print(i)
            dfNbRules = pd.read_csv(p + str(i) + '/NbRules/' + str(nbIter - 1) + '.csv', index_col=0)
            dfCoverages = pd.read_csv(p + str(i) + '/Coverages.csv', index_col=0)
            # dfCoverages = dfCoverages[dfCoverages['i']==float(nbRepeat-1)]
            for nameIndex in range(len(algName)):
                data.append([algName[nameIndex], float(dfNbRules.loc[dfNbRules['algorithm'] == algName[nameIndex]]['nbRules']),float(
                    dfCoverages.loc[dfCoverages['algorithm'] == algName[nameIndex]].head(1)['coverages'])])
        df = pd.DataFrame(data, columns=['algorithm', 'nbRules','coverages'])
        df = df.sort_values(by=['nbRules'], ascending=False)
        coverages = df.groupby(['algorithm'])
        coverages = coverages['coverages'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)
        coverages = coverages.rename(columns={'mean':'covMean','std':'covStd'})
        nbRules = df.groupby(['algorithm'])
        nbRules = nbRules['nbRules'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)
        nbRules = nbRules.rename(columns={'mean': 'nbRulesMean', 'std': 'nbRulesStd'})
        df = pd.concat([coverages,nbRules],axis=1)
        df.reset_index(level=0, inplace=True)
        df = self.ChangeAlgoNames(df)
        fig = plt.figure(figsize=(15, 15))
        ax = sns.scatterplot(x='nbRulesMean', y='covMean', hue='algorithm', style='algorithm',data=df)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        if self.save:
            fig.savefig(self.path+'GraphNBRulesVsCoverages' + ".png")


    def GraphSCCVsCoverage(self,algName,p,graphType,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []

        for i in range(nbRepeat):
            print(i)
            dfCoverages = pd.read_csv(p + str(i) + '/Coverages.csv', index_col=0)
            # dfCoverages = dfCoverages[dfCoverages['i'] == float(nbRepeat - 1)]
            dfScores = pd.read_csv(p + str(i) + '/LeaderBoard/'+ str(nbIter - 1)+'.csv', index_col=0)
            for nameIndex in range(len(algName)):
                data.append([algName[nameIndex], float(dfCoverages.loc[dfCoverages['algorithm'] == algName[nameIndex]].head(1)['coverages']),float(
                    dfScores.loc[dfScores['algorithm'] == algName[nameIndex]]['support']),float(
                    dfScores.loc[dfScores['algorithm'] == algName[nameIndex]]['confidence']),float(
                    dfScores.loc[dfScores['algorithm'] == algName[nameIndex]]['cosine'])])
        df = pd.DataFrame(data, columns=['algorithm', 'coverages','support','confidence','cosine'])
        df = df.sort_values(by=['coverages'], ascending=False)

        support = df.groupby(['algorithm'])
        support = support['support'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)

        support  = support.rename(columns={'mean':'supportMean','std':'supportStd'})

        confidence = df.groupby(['algorithm'])
        confidence = confidence['confidence'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)

        confidence = confidence.rename(columns={'mean': 'confidenceMean', 'std': 'confidenceStd'})

        cosine = df.groupby(['algorithm'])
        cosine = cosine['cosine'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)

        cosine = cosine.rename(columns={'mean': 'cosineMean', 'std': 'cosineStd'})

        coverages = df.groupby(['algorithm'])
        coverages = coverages['coverages'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)
        coverages = coverages.rename(columns={'mean': 'coveragesMean', 'std': 'coveragesStd'})

        df = pd.concat([support,confidence,cosine,coverages],axis=1)
        df.reset_index(level=0, inplace=True)
        df = self.ChangeAlgoNames(df)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
        ax = sns.scatterplot(ax=axes[0],x='coveragesMean', y='supportMean', hue='algorithm', style='algorithm',data=df)
        ax.get_legend().remove()
        ax =sns.scatterplot(ax=axes[1], x='coveragesMean', y='confidenceMean', hue='algorithm', style='algorithm', data=df)
        ax.get_legend().remove()
        ax =sns.scatterplot(ax=axes[2], x='coveragesMean', y='cosineMean', hue='algorithm', style='algorithm', data=df)
        ax.get_legend().remove()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        if self.save:
            fig.savefig(self.path+'GraphCoveragesVsSCC' + ".png")

    def GraphSCCVsNBRules(self,algName,p,graphType,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
        data = []

        for i in range(nbRepeat):
            print(i)
            dfNbRules = pd.read_csv(p + str(i) + '/NbRules/' + str(nbIter - 1) + '.csv', index_col=0)
            dfScores = pd.read_csv(p + str(i) + '/LeaderBoard/'+ str(nbIter - 1)+'.csv', index_col=0)
            for nameIndex in range(len(algName)):
                data.append([algName[nameIndex], float(dfNbRules.loc[dfNbRules['algorithm'] == algName[nameIndex]]['nbRules']),float(
                    dfScores.loc[dfScores['algorithm'] == algName[nameIndex]]['support']),float(
                    dfScores.loc[dfScores['algorithm'] == algName[nameIndex]]['confidence']),float(
                    dfScores.loc[dfScores['algorithm'] == algName[nameIndex]]['cosine'])])
        df = pd.DataFrame(data, columns=['algorithm', 'nbRules','support','confidence','cosine'])
        df = df.sort_values(by=['nbRules'], ascending=False)

        support = df.groupby(['algorithm'])
        support = support['support'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)

        support  = support.rename(columns={'mean':'supportMean','std':'supportStd'})

        confidence = df.groupby(['algorithm'])
        confidence = confidence['confidence'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)

        confidence = confidence.rename(columns={'mean': 'confidenceMean', 'std': 'confidenceStd'})

        cosine = df.groupby(['algorithm'])
        cosine = cosine['cosine'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)

        cosine = cosine.rename(columns={'mean': 'cosineMean', 'std': 'cosineStd'})

        nbRules = df.groupby(['algorithm'])
        nbRules = nbRules['nbRules'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False)
        nbRules = nbRules.rename(columns={'mean': 'nbRulesMean', 'std': 'nbRulesStd'})

        df = pd.concat([support,confidence,cosine,nbRules],axis=1)
        df.reset_index(level=0, inplace=True)
        df = self.ChangeAlgoNames(df)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        ax = sns.scatterplot(ax=axes[0],x='nbRulesMean', y='supportMean', hue='algorithm', style='algorithm',data=df)
        ax.get_legend().remove()
        ax =sns.scatterplot(ax=axes[1], x='nbRulesMean', y='confidenceMean', hue='algorithm', style='algorithm', data=df)
        ax.get_legend().remove()
        ax =sns.scatterplot(ax=axes[2], x='nbRulesMean', y='cosineMean', hue='algorithm', style='algorithm', data=df)
        ax.get_legend().remove()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        if self.save:
            fig.savefig(self.path+'GraphNBRulesVsSCC' + ".png")

    def GraphPopDistances(self,algName,p,graphType,nbIter):
        plt.cla()
        plt.clf()
        nbRepeat = len(os.listdir(p)) - 2
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
                    distances = df[df['algorithm'] == algName[nameIndex]][nameCol[0]]
                    data.append([algName[nameIndex],j,float(distances)])
        df = pd.DataFrame(data, columns=['algorithm', 'iter', 'distances'])
        df = self.ChangeAlgoNames(df)
        objectiveName = 'distances'
        fig = plt.figure(figsize=(15, 15))
        ax = sns.lineplot(x='iter', y='distances', hue='algorithm', style='algorithm', data=df)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        dfTemp = df[df['iter'] == nbIter-1].groupby(['algorithm'])
        if self.save:
            fig.savefig(self.path+'distances' + ".png")
        plt.close()
        print('distances :')
        print(dfTemp['distances'].agg(
            ['mean', 'std']).sort_values(by=['mean'], ascending=False))

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
        df.reset_index(level=0, inplace=True)
        df = self.ChangeAlgoNames(df)
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


    def GraphFitness(self,p):
        df = pd.read_csv(p)
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(15, 15))
        ax = sns.lineplot(x='iter', y='fitness', hue='algorithm', style='algorithm',markers=True, data=df)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if self.display:
            plt.show()
        else:
            plt.close(fig)
        if self.save:
            fig.savefig(self.path + ".png")









