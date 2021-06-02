from src.Algorithm.MOPSO import *
from src.Algorithm.MOBARM import *
from src.Algorithm.NSGAII import *
from src.Algorithm.MOCSOARM import *
from src.Algorithm.HMOFAARM import *
from src.Algorithm.MOSAARM import *
from src.Algorithm.MOWSAARM import *
from src.Algorithm.MOCatSOARM import *
from src.Algorithm.MOTLBOARM import *
from src.Algorithm.MOFPAARM import *
from src.Algorithm.MOALOARM import *
from src.Algorithm.MODAARM import *
from src.Algorithm.MOHSBOTSARM import *
from src.Algorithm.MODEARM import *
from src.Algorithm.NSHSDEARM import *
from src.Algorithm.MOGEAARM import *
from src.Algorithm.MOGSAARM import *
from src.Algorithm.MOSSOARM import *
from src.Algorithm.MOWOAARM import *
from src.Algorithm.MOSOSARM import *
from os import path,mkdir


class Experiment:
    def __init__(self,algListNames,objectiveNames,criterionList,data,populationSize,nbIteration,nbRepetition,display=False,path='Experiments/'):
        self.algListNames = algListNames
        self.objectiveNames = objectiveNames
        self.criterionList = criterionList
        self.populationSize = populationSize
        self.nbIteration = nbIteration
        self.nbRepetition = nbRepetition
        self.data = data
        self.algList = []
        self.display = display
        self.path = path
        self.CheckIfFolderExist(self.path)
        self.perf = Performances(algListNames, criterionList, objectiveNames)

    def CheckIfFolderExist(self,p):
        if(not path.exists(p)):
            mkdir(p)

    def InitAlgList(self):
        for name in self.algListNames:
            if name == 'mocsoarm':
                self.algList.append(MOCSOARM(self.data.shape[1],self.populationSize,self.nbIteration,len(self.objectiveNames),self.objectiveNames,self.data))
            if name == 'mopso':
                self.algList.append(
                    MOPSO(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mobarm':
                self.algList.append(
                    MOBARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'nsgaii':
                self.algList.append(
                    NSGAII(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'hmofaarm':
                self.algList.append(
                    HMOFAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mosaarm':
                self.algList.append(
                    MOSAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mowsaarm':
                self.algList.append(
                    MOWSAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mocatsoarm':
                self.algList.append(
                    MOCatSOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'motlboarm':
                self.algList.append(
                    MOTLBOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mofpaarm':
                self.algList.append(
                    MOFPAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'moaloarm':
                self.algList.append(
                    MOALOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'modaarm':
                self.algList.append(
                    MODAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mohsbotsarm':
                self.algList.append(
                    MOHSBOTSARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'modearm':
                self.algList.append(
                    MODEARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'nshsdearm':
                self.algList.append(
                    NSHSDEARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mogeaarm':
                self.algList.append(
                    MOGEAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mogsaarm':
                self.algList.append(
                    MOGSAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mossoarm':
                self.algList.append(
                    MOSSOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mowoaarm':
                self.algList.append(
                    MOWOAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mososarm':
                self.algList.append(
                    MOSOSARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))


    def Run(self):
        graphPath = self.path+'Graphs/'
        self.CheckIfFolderExist(graphPath)
        scoreGraphPath = graphPath + 'Scores/'
        nbRulesGraphPath = graphPath + 'NbRules/'
        self.CheckIfFolderExist(scoreGraphPath)
        self.CheckIfFolderExist(nbRulesGraphPath)
        for rep in range(self.nbRepetition):
            self.InitAlgList()
            executionTimeGraphPath = graphPath + 'ExecutionTime/'
            scoreGraphPath = graphPath + 'Scores/'+str(rep)+'/'
            nbRulesGraphPath = graphPath + 'NbRules/'+str(rep)+'/'
            self.CheckIfFolderExist(executionTimeGraphPath)
            self.CheckIfFolderExist(scoreGraphPath)
            self.CheckIfFolderExist(nbRulesGraphPath)
            self.CheckIfFolderExist(self.path+str(rep)+'/')
            for i in range(self.nbIteration):
                k = 0
                for alg in self.algList:
                    alg.Run(self.data, i)
                    alg.fitness.GetParetoFront()
                    self.perf.UpdatePerformances(score=alg.fitness.paretoFront, executionTime=alg.executionTime, i=i,
                                            algorithmName=self.algListNames[k])
                    print(self.algListNames[k])
                    print(alg.fitness.paretoFront)
                    k += 1
                graph = Graphs(self.objectiveNames, self.perf.scores, path=scoreGraphPath + str(i), display=self.display)
                graph.GraphScores()
                graph = Graphs(self.objectiveNames, self.perf.nbRules, path=nbRulesGraphPath+str(i), display=self.display)
                graph.GraphNbRules()
                self.perf.UpdateLeaderBoard()
                self.perf.SaveIntermediaryPerf(self.path+str(rep)+'/',i)
                self.perf.FreeScores()
            graph = Graphs(['execution Time'], self.perf.executionTime, path=executionTimeGraphPath+str(rep), display=self.display)
            graph.GraphExecutionTime()
            self.perf.SaveFinalPerf(self.path+str(rep)+'/')
