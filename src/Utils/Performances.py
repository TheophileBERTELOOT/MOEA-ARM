import pandas as pd
from src.Utils.Graphs import *
from src.Utils.Fitness import *
from os import path,mkdir
class Performances:
    def __init__(self,algorithmList,criterionList,nbItem,objectiveNames=[]):
        self.algorithmList = algorithmList
        self.criterionList = criterionList
        self.objectiveNames = objectiveNames
        self.leaderBoard = np.zeros((len(algorithmList),len(objectiveNames)),dtype=float)
        self.nbItem = nbItem
        self.Init()

    def InitScores(self):
        self.columnsScores = ['algorithm'] + self.objectiveNames
        self.scores = pd.DataFrame(columns=self.columnsScores)

    def InitNbRules(self):
        self.nbRules = pd.DataFrame(columns=['algorithm', 'nbRules'])

    def InitDistances(self):
        self.distances = pd.DataFrame(columns=['algorithm','distances'])

    def InitPopDistances(self):
        self.popDistances = pd.DataFrame(columns=['algorithm', 'distances'])

    def InitCoverages(self):
        self.coverages = pd.DataFrame(columns=['algorithm','coverages'])

    def InitExecutionTime(self):
        self.columnsET = ['i', 'algorithm', 'execution Time']
        self.executionTime = pd.DataFrame(columns=self.columnsET)

    def InitTestedIndividuals(self):

        self.transformTestedIndividuals =  pd.DataFrame(columns=['algorithm','x','y'])
        print(self.transformTestedIndividuals)
        self.testedIndividuals = pd.DataFrame(columns=['algorithm']+[str(i) for i in range(self.nbItem*2)])


    def Init(self):
        if 'scores' in self.criterionList:
            self.InitScores()
            self.InitNbRules()
        if 'execution time' in self.criterionList:
            self.InitExecutionTime()
        if 'distances' in self.criterionList:
            self.InitDistances()
        if 'popDistance' in self.criterionList:
            self.InitPopDistances()
        if 'coverages' in self.criterionList:
            self.InitCoverages()
        if 'tested' in self.criterionList:
            self.InitTestedIndividuals()

    def Free(self):
        self.InitScores()
        self.InitCoverages()
        self.InitDistances()
        self.InitPopDistances()
        self.InitNbRules()


    def UpdatePerformances(self,score = [],executionTime=[],popDistance=[],testedIndividuals=[],testedScoreIndividuals=[],coverage=0,distance=0,i=0,algorithmName=''):
        if 'scores' in self.criterionList:
            score = [[algorithmName]+list(score[i]) for i in range(len(score))]
            nbRule = pd.DataFrame([[algorithmName,len(score)]],columns=['algorithm','nbRules'])
            scoreDF = pd.DataFrame(score,columns=self.columnsScores)
            self.scores = self.scores.append(scoreDF, ignore_index=True)
            self.nbRules = self.nbRules.append(nbRule,ignore_index=True)
            self.nbRules = self.nbRules.sort_values(by=['nbRules'], ascending=False)
        if 'execution time' in self.criterionList:
            executionTimeDF = pd.DataFrame([[i,algorithmName,executionTime]],columns=self.columnsET)
            self.executionTime = self.executionTime.append(executionTimeDF, ignore_index=True)
        if 'distances' in self.criterionList:
            distanceDF = pd.DataFrame([[ algorithmName, distance]], columns=['algorithm', 'distances'])
            self.distances = self.distances.append(distanceDF, ignore_index=True)
            self.distances = self.distances.sort_values(by=['distances'], ascending=False)
        if 'popDistance'in self.criterionList:
            distanceDF = pd.DataFrame([[algorithmName, popDistance]], columns=['algorithm', 'distances'])
            self.popDistances = self.popDistances.append(distanceDF, ignore_index=True)
            self.popDistances = self.popDistances.sort_values(by=['distances'], ascending=False)
        if 'coverages' in self.criterionList:
            coverageDF = pd.DataFrame([[ algorithmName, coverage]], columns=[ 'algorithm', 'coverages'])
            self.coverages = self.coverages.append(coverageDF, ignore_index=True)
            self.coverages = self.coverages.sort_values(by=['coverages'], ascending=False)
        if 'tested' in self.criterionList:
            if (i == 0 or i == 1 or i == 10 or i == 20 or i == 30 or i == 40 or i == 49):
                testedData = []
                for i in range(len(testedIndividuals)):
                    testedData.append([algorithmName]+[testedIndividuals[i][j] for j in range(len(testedIndividuals[i]))])
                testedDf = pd.DataFrame(testedData,columns=['algorithm']+[str(i) for i in range(self.nbItem*2)])
                self.testedIndividuals = self.testedIndividuals.append(testedDf, ignore_index=True)




    def UpdateLeaderBoard(self):
        self.leaderBoard = np.array([[0 for j in range(len(self.objectiveNames))] for i in range(len(self.algorithmList))],dtype=float)
        for i in range(len(self.algorithmList)):
            solutionsi = self.scores[self.scores['algorithm'] == self.algorithmList[i]][self.objectiveNames].to_numpy()
            for j in range(len(solutionsi)):
                for k in range(len(self.objectiveNames)):
                    self.leaderBoard[i][k] = self.leaderBoard[i][k] + solutionsi[j][k]
            self.leaderBoard[i] = self.leaderBoard[i]/len(solutionsi)
            self.leaderBoard[i] = np.round(self.leaderBoard[i],2)
        self.leaderBoardSorted = list(zip( self.algorithmList,self.leaderBoard[:,0],self.leaderBoard[:,1],self.leaderBoard[:,2]))
        self.leaderBoardSorted = np.array(sorted(self.leaderBoardSorted, key=lambda x: x[1],reverse=True), dtype="object")

        if 'scores' in self.criterionList:
            print(self.leaderBoardSorted)



    def RefactorTSNE(self):
        data = self.testedIndividuals.drop('algorithm',axis=1)
        algorithms = self.testedIndividuals['algorithm']
        data = TSNE(n_components=2, learning_rate='auto',
                        init='random').fit_transform(np.asarray(data,dtype='float64'))
        transformed = pd.DataFrame(list(zip(list(algorithms),data[:,0],data[:,1])),columns=['algorithm','x','y'])
        transformed = transformed.drop_duplicates()
        self.transformTestedIndividuals = transformed
        print(self.transformTestedIndividuals)




    def SaveIntermediaryPerf(self,p,i,updating=False):
        scorePath = p+'Score/'
        nbRulesPath = p+'NbRules/'
        leaderBoardPath = p+'LeaderBoard/'
        popDistancePath = p+'PopDistance/'
        testedIndividualsPath = p+'TestedIndividuals/'
        testedScoreIndividualsPath = p + 'TestedScoreIndividuals/'
        if (not path.exists(scorePath)):
            mkdir(scorePath)
        if (not path.exists(nbRulesPath)):
            mkdir(nbRulesPath)
        if (not path.exists(leaderBoardPath)):
            mkdir(leaderBoardPath)
        if (not path.exists(popDistancePath)):
            mkdir(popDistancePath)
        if (not path.exists(testedIndividualsPath)):
            mkdir(testedIndividualsPath)
        if (not path.exists(testedScoreIndividualsPath)):
            mkdir(testedScoreIndividualsPath)
        if updating:
            scores = pd.read_csv(p+'Score/'+str(i)+'.csv',index_col=0,header=0)
            index = scores['algorithm'] == self.algorithmList[0]
            scores.loc[index,['algorithm','support','confidence','cosine']] = list(self.scores.loc[0])
            scores.to_csv(p+'Score/'+str(i)+'.csv',index=True)
            nbRules = pd.read_csv(p + 'NbRules/' + str(i) + '.csv',index_col=0,header=0)
            index = nbRules['algorithm'] == self.algorithmList[0]
            nbRules.loc[index,['algorithm','nbRules']] = list(self.nbRules.loc[0])
            nbRules.to_csv(p + 'NbRules/' + str(i) + '.csv',index=True)
            leaderBoard = pd.read_csv(p+'LeaderBoard/'+str(i)+'.csv',index_col=0,header=0)
            index = leaderBoard['algorithm'] == self.algorithmList[0]
            df = pd.DataFrame(self.leaderBoardSorted,columns=['algorithm']+[obj for obj in self.objectiveNames])
            leaderBoard.loc[index,['algorithm','support','confidence','cosine']] = list(df.loc[0])
            leaderBoard.to_csv(p+'LeaderBoard/'+str(i)+'.csv',index=True)
        else:
            self.scores.to_csv(p+'Score/'+str(i)+'.csv')
            self.nbRules.to_csv(p+'NbRules/'+str(i)+'.csv')
            if 'popDistances' in self.objectiveNames:
                self.popDistances.to_csv(p+'PopDistance/'+str(i)+'.csv')
            if (i == 0 or i == 1 or i == 10 or i == 20 or i == 30 or i == 40 or i == 49):
                self.RefactorTSNE()
                self.testedIndividuals.to_csv(p+'TestedIndividuals/'+str(i)+'.csv')



    def SaveFinalPerf(self,p,updating=False):
        if (not path.exists(p)):
            mkdir(p)
        if updating:
            executionTime = pd.read_csv(p+'ExecutionTime.csv',index_col=0,header=0)
            for _ in range(len(self.executionTime)):
                index = (executionTime['algorithm'] == self.algorithmList[0]) & (executionTime['i'] == _)
                executionTime.loc[index, ['i', 'algorithm', 'execution Time']] = list(self.executionTime.loc[_])


            executionTime.to_csv(p + 'ExecutionTime.csv',index=True)

            distances = pd.read_csv(p+'Distances.csv',header=0,index_col=0,)
            index = distances['algorithm'] == self.algorithmList[0]
            distances.loc[index,['algorithm','distances']] = list(self.distances.loc[0])
            distances.to_csv(p + 'Distances.csv',index=True)

            coverages = pd.read_csv(p + 'Coverages.csv',header=0,index_col=0,)
            index = coverages['algorithm'] == self.algorithmList[0]
            coverages.loc[index,['algorithm','coverages']] = list(self.coverages.loc[0])
            coverages.to_csv(p + 'Coverages.csv',index=True)
        else:
            self.executionTime.to_csv(p+'ExecutionTime.csv')
            self.distances.to_csv(p + 'Distances.csv')
            self.coverages.to_csv(p + 'Coverages.csv')
            print("end of the run")
        self.InitExecutionTime()

    def CalculFitness(self,p,nbIter):
        # datasets = os.listdir(p)
        datasets = ['CONGRESS','BRIDGES','FLAG']
        nbDataset = len(datasets)
        f = open(p + 'fitness.csv', 'w')
        f.write('algorithm,dataset,iter,fitness\n')
        for dataset in datasets:

            dp = p+dataset+'/'
            fitness = np.zeros((len(self.algorithmList), nbIter), dtype=float)
            nbRepeat = len(os.listdir(dp)) - 2
            for r in range(nbRepeat):
                print(r)
                dr = dp + str(r)+'/'
                coverages = pd.read_csv(dr+'Coverages.csv')
                distances = pd.read_csv(dr+'Distances.csv')
                execTimes = pd.read_csv(dr+'ExecutionTime.csv')
                for i in range(nbIter):
                    leaderboard =  pd.read_csv(dr+'LeaderBoard/'+str(i)+'.csv')
                    nbRules = pd.read_csv(dr+'NbRules/'+str(i)+'.csv')
                    coverage = coverages[coverages['i'] == i]
                    distance = distances[distances['i'] == i]
                    execTime = execTimes[execTimes['i'] == i]
                    sumCos = sum(leaderboard['cosine'])
                    sumSupp = sum(leaderboard['support'])
                    sumConf = sum(leaderboard['confidence'])
                    sumNbRule = sum(nbRules['nbRules'])
                    sumCoverage = sum(coverage['coverages'])
                    sumDistance = sum(distance['distances'])
                    sumExecTime = sum(execTime['execution Time'])
                    for algID in range(len(self.algorithmList)):
                        alg = self.algorithmList[algID]
                        algCos = float(leaderboard[leaderboard['algorithm'] == alg]['cosine'])
                        algSupp = float(leaderboard[leaderboard['algorithm'] == alg]['support'])
                        algConf = float(leaderboard[leaderboard['algorithm'] == alg]['confidence'])
                        algNbRule = float(nbRules[nbRules['algorithm'] == alg]['nbRules'])
                        algCoverage = float(coverage[coverage['algorithm'] == alg]['coverages'])
                        algDistance = float(distance[distance['algorithm'] == alg]['distances'])
                        algExecTime = float(execTime[execTime['algorithm'] == alg]['execution Time'])
                        n = 7
                        fitness[algID][i] = fitness[algID][i] + ((algCos/sumCos) +\
                        (algSupp / sumSupp) + \
                        (algConf / sumConf) + \
                        (algNbRule / sumNbRule) + \
                        (algCoverage / sumCoverage) + \
                        (algDistance / sumDistance) + \
                        (1-(algExecTime/sumExecTime)))
            fitness = fitness/nbRepeat

            for i in range(len(self.algorithmList)):
                alg = self.algorithmList[i]
                for j in range(nbIter):
                    f.write(alg+','+dataset+','+str(j)+','+str(np.round(fitness[i][j],2))+'\n')
        f.close()


    def CalcCatchup(self,alg,measure,p,repeat,dataset,nbIter):
        catchup = 0
        dp = p + dataset + '/'
        dr = dp + str(repeat) + '/'
        leaderboard = pd.read_csv(dr + 'LeaderBoard/' + str(nbIter-1) + '.csv')
        threshold = float(leaderboard[leaderboard['algorithm'] == 'custom'][measure])
        for i in range(nbIter-3):
            score = float(leaderboard[leaderboard['algorithm'] == alg][measure])
            if score > threshold:
                return catchup
            else:
                catchup += 1
        return catchup


    def CalculPerf(self, p, nbIter):

        # datasets = os.listdir(p)
        datasets = ['IRIS','FLAG','ABALONE_C','CRX_C','MUSHROOM']


        nbDataset = len(datasets)
        f = open(p + 'results.csv', 'w')
        f.write('algorithm,dataset,coverages,stdCoverage,distances,stdDistances,support,stdSupport,confidence,stdConfidence,'
                'cosine,stdCosine,nbRules,stdNbRules,Execution Time,std Execution Time,suppCatchup,stdSuppCatchup,'
                'confCatchup,stdConfCatchup,cosCatchup,stdCosCatchup\n')
        for datasetIndex in range(len(datasets)):
            dataset = datasets[datasetIndex]
            dp = p + dataset + '/'
            nbRepeat = len(os.listdir(dp)) - 2
            algCos = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algSupp = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algConf = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algNbRule = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algCoverage = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algDistance = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algExecTime = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algCosCatch = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algConfCatch = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            algSuppCatch = np.zeros((len(self.algorithmList), nbRepeat), dtype=float)
            for r in range(nbRepeat):
                print(r)
                print(dataset)
                dr = dp + str(r) + '/'
                if dataset == 'MUSHROOM':
                    coverages = pd.read_csv(dr + 'Coverages.csv')
                    coverages = coverages[coverages['i'] == nbIter -1]
                    distances = pd.read_csv(dr + 'Distances.csv')
                    distances = distances[distances['i'] == nbIter - 1]

                else:
                    coverages = pd.read_csv(dr + 'Coverages.csv')
                    distances = pd.read_csv(dr + 'Distances.csv')

                execTimes = pd.read_csv(dr + 'ExecutionTime.csv')

                leaderboard = pd.read_csv(dr + 'LeaderBoard/' + str(nbIter-1) + '.csv')

                nbRules = pd.read_csv(dr + 'NbRules/' + str(nbIter-1) + '.csv')

                execTime = execTimes[execTimes['i'] == nbIter-1]

                for algID in range(len(self.algorithmList)):
                    alg = self.algorithmList[algID]

                    algCos[algID][r] = float(leaderboard[leaderboard['algorithm'] == alg]['cosine'])
                    algCosCatch[algID][r] = self.CalcCatchup(alg, 'cosine', p, r, dataset, nbIter)
                    algSupp[algID][r] = float(leaderboard[leaderboard['algorithm'] == alg]['support'])
                    algSuppCatch[algID][r] = self.CalcCatchup(alg,'support',p,r,dataset,nbIter)
                    algConf[algID][r] = float(leaderboard[leaderboard['algorithm'] == alg]['confidence'])
                    algConfCatch[algID][r] = self.CalcCatchup(alg, 'confidence', p, r, dataset, nbIter)
                    algNbRule[algID][r] = float(nbRules[nbRules['algorithm'] == alg]['nbRules'])
                    algCoverage[algID][r] = float(coverages[coverages['algorithm'] == alg]['coverages'])
                    algDistance[algID][r] = float(distances[distances['algorithm'] == alg]['distances'])
                    algExecTime[algID][r] = float(execTime[execTime['algorithm'] == alg]['execution Time'])

            for i in range(len(self.algorithmList)):
                alg = self.algorithmList[i]
                f.write(alg + ',' + dataset +  ','  +str(np.round(np.mean(algCoverage[i]),2))+ ',' +str(np.round(np.std(algCoverage[i]),2))
                        + ',' +str(np.round(np.mean(algDistance[i]),2))+ ',' +str(np.round(np.std(algDistance[i]),2))
                        + ',' +str(np.round(np.mean(algSupp[i]),2))+ ','+str(np.round(np.std(algSupp[i]),2))+ ','
                        +str(np.round(np.mean(algConf[i]),2))+',' +str(np.round(np.std(algConf[i]),2))+','+
                        str(np.round(np.mean(algCos[i]),2))+','+str(np.round(np.std(algCos[i]),2))+
                        ',' +str(np.round(np.mean(algNbRule[i]),2))+',' +str(np.round(np.std(algNbRule[i]),2))+','
                        +str(np.round(np.mean(algExecTime[i]),2))+','+str(np.round(np.std(algExecTime[i]),2))+
                        ',' +str(np.round(np.mean(algSuppCatch[i]),2))+',' +str(np.round(np.std(algSuppCatch[i]),2))+
                        ',' +str(np.round(np.mean(algConfCatch[i]),2))+',' +str(np.round(np.std(algConfCatch[i]),2))+
                        ',' +str(np.round(np.mean(algCosCatch[i]),2))+',' +str(np.round(np.std(algCosCatch[i]),2))+'\n')
        f.close()




