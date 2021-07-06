import pandas as pd
from src.Utils.Graphs import *
from src.Utils.Fitness import *
from os import path,mkdir
class Performances:
    def __init__(self,algorithmList,criterionList,objectiveNames=[]):
        self.algorithmList = algorithmList
        self.criterionList = criterionList
        self.objectiveNames = objectiveNames
        self.leaderBoard = np.zeros((len(algorithmList),len(objectiveNames)),dtype=float)
        self.Init()

    def InitScores(self):
        self.columnsScores = ['algorithm'] + self.objectiveNames
        self.scores = pd.DataFrame(columns=self.columnsScores)

    def InitNbRules(self):
        self.nbRules = pd.DataFrame(columns=['algorithm', 'nbRules'])

    def InitDistances(self):
        self.distances = pd.DataFrame(columns=['algorithm','distances'])

    def InitCoverages(self):
        self.coverages = pd.DataFrame(columns=['algorithm','coverages'])

    def InitExecutionTime(self):
        self.columnsET = ['i', 'algorithm', 'execution Time']
        self.executionTime = pd.DataFrame(columns=self.columnsET)

    def Init(self):
        if 'scores' in self.criterionList:
            self.InitScores()
            self.InitNbRules()
        if 'execution time' in self.criterionList:
            self.InitExecutionTime()
        if 'distances' in self.criterionList:
            self.InitDistances()
        if 'coverages' in self.criterionList:
            self.InitCoverages()

    def Free(self):
        self.InitScores()
        self.InitDistances()
        self.InitCoverages()
        self.InitNbRules()

    def UpdatePerformances(self,score = [],executionTime=[],coverage=0,distance=0,i=0,algorithmName=''):
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
            distanceDF = pd.DataFrame([[algorithmName,distance]], columns=['algorithm','distances'])
            self.distances = self.distances.append(distanceDF, ignore_index=True)
            self.distances = self.distances.sort_values(by=['distances'], ascending=False)
        if 'coverages' in self.criterionList:
            coverageDF = pd.DataFrame([[algorithmName,coverage]], columns=['algorithm','coverages'])
            self.coverages = self.coverages.append(coverageDF, ignore_index=True)
            self.coverages = self.coverages.sort_values(by=['coverages'], ascending=False)

    def UpdateLeaderBoard(self):
        self.leaderBoard = np.array([[0 for j in range(len(self.objectiveNames))] for i in range(len(self.algorithmList))],dtype=float)
        fitness = Fitness('horizontal_binary',self.objectiveNames,0)
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
            print(self.scores)
            print(self.leaderBoardSorted)
        if 'distances' in self.criterionList:
            print(self.distances)
        if 'coverages' in self.criterionList:
            print(self.coverages)





    def SaveIntermediaryPerf(self,p,i):
        scorePath = p+'Score/'
        nbRulesPath = p+'NbRules/'
        leaderBoardPath = p+'LeaderBoard/'
        if (not path.exists(scorePath)):
            mkdir(scorePath)
        if (not path.exists(nbRulesPath)):
            mkdir(nbRulesPath)
        if (not path.exists(leaderBoardPath)):
            mkdir(leaderBoardPath)
        self.scores.to_csv(p+'Score/'+str(i)+'.csv')
        self.nbRules.to_csv(p+'NbRules/'+str(i)+'.csv')

        pd.DataFrame(self.leaderBoardSorted,columns=['algorithm']+[obj for obj in self.objectiveNames]).to_csv(p+'LeaderBoard/'+str(i)+'.csv')


    def SaveFinalPerf(self,p):
        if (not path.exists(p)):
            mkdir(p)
        self.executionTime.to_csv(p+'ExecutionTime.csv')
        self.distances.to_csv(p + 'Distances.csv')
        self.coverages.to_csv(p + 'Coverages.csv')
        self.InitExecutionTime()





