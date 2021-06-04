import pandas as pd
from src.Utils.Graphs import *
from src.Utils.Fitness import *
from os import path,mkdir
class Performances:
    def __init__(self,algorithmList,criterionList,objectiveNames=[]):
        self.algorithmList = algorithmList
        self.criterionList = criterionList
        self.objectiveNames = objectiveNames
        self.leaderBoard = np.zeros(len(algorithmList),dtype=float)
        self.Init()

    def InitScores(self):
        self.columnsScores = ['algorithm'] + self.objectiveNames
        self.nbRules = pd.DataFrame(columns=['algorithm','nbRules'])
        self.scores = pd.DataFrame(columns=self.columnsScores)

    def InitExecutionTime(self):
        self.columnsET = ['i', 'algorithm', 'execution Time']
        self.executionTime = pd.DataFrame(columns=self.columnsET)

    def Init(self):
        if 'scores' in self.criterionList:
            self.InitScores()
        if 'execution time' in self.criterionList:
            self.InitExecutionTime()

    def FreeScores(self):
        self.InitScores()

    def UpdatePerformances(self,score = [],executionTime=[],i=0,algorithmName=''):
        if 'scores' in self.criterionList:
            score = [[algorithmName]+list(score[i]) for i in range(len(score))]
            nbRule = pd.DataFrame([[algorithmName,len(score)]],columns=['algorithm','nbRules'])
            scoreDF = pd.DataFrame(score,columns=self.columnsScores)
            self.scores = self.scores.append(scoreDF, ignore_index=True)
            self.nbRules = self.nbRules.append(nbRule,ignore_index=True)
        if 'execution time' in self.criterionList:
            executionTimeDF = pd.DataFrame([[i,algorithmName,executionTime]],columns=self.columnsET)
            self.executionTime = self.executionTime.append(executionTimeDF, ignore_index=True)

    def UpdateLeaderBoard(self):
        self.leaderBoard = np.array([0 for i in range(len(self.algorithmList))],dtype=float)
        fitness = Fitness('horizontal_binary',self.objectiveNames,0)
        for i in range(len(self.algorithmList)):
            solutionsi = self.scores[self.scores['algorithm'] == self.algorithmList[i]][self.objectiveNames].to_numpy()
            for j in range(len(solutionsi)):
                self.leaderBoard[i]+=sum(solutionsi[j])
            self.leaderBoard[i] = np.round(self.leaderBoard[i],2)
        self.leaderBoardSorted = list(zip(self.leaderBoard, self.algorithmList))
        self.leaderBoardSorted = np.array(sorted(self.leaderBoardSorted, key=lambda x: x[0],reverse=True), dtype="object")
        print(self.scores)
        print(self.leaderBoardSorted)

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
        pd.DataFrame(self.leaderBoardSorted,columns=['score','algorithm']).to_csv(p+'LeaderBoard/'+str(i)+'.csv')


    def SaveFinalPerf(self,p):
        if (not path.exists(p)):
            mkdir(p)
        self.executionTime.to_csv(p+'ExecutionTime.csv')
        self.InitExecutionTime()





