import pandas as pd
from src.Utils.Graphs import *
class Performances:
    def __init__(self,algorithmList,criterionList,objectiveNames=[]):
        self.algorithmList = algorithmList
        self.criterionList = criterionList
        self.objectiveNames = objectiveNames
        self.Init()

    def InitScores(self):
        self.columnsScores = ['algorithm'] + self.objectiveNames
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
            scoreDF = pd.DataFrame(score,columns=self.columnsScores)
            self.scores = self.scores.append(scoreDF, ignore_index=True)
        if 'execution time' in self.criterionList:
            executionTimeDF = pd.DataFrame([[i,algorithmName,executionTime]],columns=self.columnsET)
            self.executionTime = self.executionTime.append(executionTimeDF, ignore_index=True)
