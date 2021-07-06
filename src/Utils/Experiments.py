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
from src.Algorithm.MOCSSARM import *
from src.Algorithm.Custom import *
from src.Utils.HyperParameters import *
from os import path,mkdir


class Experiment:
    def __init__(self,algListNames,objectiveNames,criterionList,data,populationSize,nbIteration,nbRepetition,iterationInitial,sizeHead=5,display=False,path='Experiments/'):
        self.algListNames = algListNames
        self.objectiveNames = objectiveNames
        self.criterionList = criterionList
        self.populationSize = populationSize
        self.nbIteration = nbIteration
        self.nbRepetition = nbRepetition
        self.iterationInitial = iterationInitial
        self.sizeHead = sizeHead
        self.hyperParametersList = {}
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
        self.algList = []
        for name in self.algListNames:
            if name == 'custom':
                self.algList.append(
                    CUSTOM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                           self.objectiveNames, self.data))
            if name == 'mocsoarm':
                h = HyperParameters(['ruthlessRatio'])
                h.LoadHyperParameters('HyperParameters/MOCSOARM/bestParameters.json')
                self.algList.append(MOCSOARM(self.data.shape[1],self.populationSize,self.nbIteration,len(self.objectiveNames),self.objectiveNames,self.data,hyperParameters=h))
            if name == 'mopso':
                h = HyperParameters(['inertie', 'localAccelaration', 'globalAcceleration'])
                h.LoadHyperParameters('HyperParameters/MOPSO/bestParameters.json')
                self.algList.append(
                    MOPSO(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mobarm':
                self.algList.append(
                    MOBARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'nsgaii':
                h = HyperParameters(['mutationRate', 'crossOverRate'])
                h.LoadHyperParameters('HyperParameters/NSGAII/bestParameters.json')
                self.algList.append(
                    NSGAII(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'hmofaarm':
                h = HyperParameters(['alpha', 'beta0', 'crossOverRate'])
                h.LoadHyperParameters('HyperParameters/HMOFAARM/bestParameters.json')
                self.algList.append(
                    HMOFAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mosaarm':
                h = HyperParameters(['alpha'])
                h.LoadHyperParameters('HyperParameters/MOSAARM/bestParameters.json')
                self.algList.append(
                    MOSAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mowsaarm':
                h = HyperParameters(['velocityFactor','enemyProb'])
                h.LoadHyperParameters('HyperParameters/MOWSAARM/bestParameters.json')
                self.algList.append(
                    MOWSAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mocatsoarm':
                h = HyperParameters(['mixtureRatio','velocityRatio'])
                h.LoadHyperParameters('HyperParameters/MOCatSOARM/bestParameters.json')
                self.algList.append(
                    MOCatSOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'motlboarm':
                self.algList.append(
                    MOTLBOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mofpaarm':
                h = HyperParameters(['P','gamma'])
                h.LoadHyperParameters('HyperParameters/MOFPAARM/bestParameters.json')
                self.algList.append(
                    MOFPAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'moaloarm':
                self.algList.append(
                    MOALOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'modaarm':
                h = HyperParameters(['s','a','c','f','e','w'])
                h.LoadHyperParameters('HyperParameters/MODAARM/bestParameters.json')
                self.algList.append(
                    MODAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mohsbotsarm':
                self.algList.append(
                    MOHSBOTSARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'modearm':
                h = HyperParameters(['F','CR'])
                h.LoadHyperParameters('HyperParameters/MODEARM/bestParameters.json')
                self.algList.append(
                    MODEARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'nshsdearm':
                h = HyperParameters(['F','Fw','PAR'])
                h.LoadHyperParameters('HyperParameters/NSHSDEARM/bestParameters.json')
                self.algList.append(
                    NSHSDEARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mogeaarm':
                h = HyperParameters(['Jr','Sr','epsilon'])
                h.LoadHyperParameters('HyperParameters/MOGEAARM/bestParameters.json')
                self.algList.append(
                    MOGEAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mogsaarm':
                h = HyperParameters(['G'])
                h.LoadHyperParameters('HyperParameters/MOGSAARM/bestParameters.json')
                self.algList.append(
                    MOGSAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mossoarm':
                h = HyperParameters(['PF'])
                h.LoadHyperParameters('HyperParameters/MOSSOARM/bestParameters.json')
                self.algList.append(
                    MOSSOARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mowoaarm':
                h = HyperParameters(['b'])
                h.LoadHyperParameters('HyperParameters/MOWOAARM/bestParameters.json')
                self.algList.append(
                    MOWOAARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data,hyperParameters=h))
            if name == 'mososarm':
                self.algList.append(
                    MOSOSARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))
            if name == 'mocssarm':
                self.algList.append(
                    MOCSSARM(self.data.shape[1], self.populationSize, self.nbIteration, len(self.objectiveNames),
                             self.objectiveNames, self.data))


    def Run(self):
        graphPath = self.path+'Graphs/'
        self.CheckIfFolderExist(graphPath)
        scoreGraphPath = graphPath + 'Scores/'
        nbRulesGraphPath = graphPath + 'NbRules/'
        coveragesGraphPath = graphPath + 'Coverages/'
        distancesGraphPath = graphPath + 'Distances/'
        rulesPath = self.path+'Rules/'
        self.CheckIfFolderExist(scoreGraphPath)
        self.CheckIfFolderExist(nbRulesGraphPath)
        self.CheckIfFolderExist(coveragesGraphPath)
        self.CheckIfFolderExist(distancesGraphPath)
        self.CheckIfFolderExist(rulesPath)
        for rep in range(self.iterationInitial,self.iterationInitial+self.nbRepetition):
            self.InitAlgList()
            executionTimeGraphPath = graphPath + 'ExecutionTime/'
            scoreGraphPath = graphPath + 'Scores/'+str(rep)+'/'
            nbRulesGraphPath = graphPath + 'NbRules/'+str(rep)+'/'
            coveragesGraphPath = graphPath + 'Coverages/'+str(rep)+'/'
            distancesGraphPath = graphPath + 'Distances/'+str(rep)+'/'
            rulesPathRep = rulesPath + str(rep)+'/'
            self.CheckIfFolderExist(executionTimeGraphPath)
            self.CheckIfFolderExist(scoreGraphPath)
            self.CheckIfFolderExist(coveragesGraphPath)
            self.CheckIfFolderExist(distancesGraphPath)
            self.CheckIfFolderExist(self.path+str(rep)+'/')
            self.CheckIfFolderExist(rulesPathRep)
            for i in range(self.nbIteration):
                k = 0
                for alg in self.algList:
                    alg.Run(self.data, i)
                    alg.population.CheckIfNull()
                    alg.fitness.ComputeScorePopulation(alg.population.population, self.data)
                    alg.fitness.GetParetoFront(alg.population)
                    #alg.fitness.GetHead(self.sizeHead,alg.population)
                    #alg.fitness.GetUniquePop(alg.population)
                    alg.fitness.GetDistances()
                    alg.fitness.GetCoverage(self.data)
                    alg.fitness.WritePop(rulesPathRep+self.algListNames[k]+'.txt')
                    self.perf.UpdatePerformances(score=alg.fitness.paretoFront, executionTime=alg.executionTime, i=i,
                                            algorithmName=self.algListNames[k],coverage=alg.fitness.coverage,distance=alg.fitness.averageDistances)
                    k += 1

                graph = Graphs(self.objectiveNames, self.perf.scores, path=scoreGraphPath + str(i), display=self.display)
                graph.GraphScores()
                self.perf.UpdateLeaderBoard()
                self.perf.SaveIntermediaryPerf(self.path+str(rep)+'/',i)
                if i<self.nbIteration-1:
                    self.perf.Free()
            graph = Graphs(['execution Time'], self.perf.executionTime, path=executionTimeGraphPath+str(rep), display=self.display)
            graph.GraphExecutionTime()
            graph = Graphs(self.objectiveNames, self.perf.nbRules, path=nbRulesGraphPath + str(i), display=self.display)
            graph.GraphNbRules()
            graph = Graphs(self.objectiveNames, self.perf.coverages, path=coveragesGraphPath + str(i), display=self.display)
            graph.GraphCoverages()
            graph = Graphs(self.objectiveNames, self.perf.distances, path=distancesGraphPath + str(i),
                           display=self.display)
            graph.GraphDistances()

            self.perf.SaveFinalPerf(self.path+str(rep)+'/')
