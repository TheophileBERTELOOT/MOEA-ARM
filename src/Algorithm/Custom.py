from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class CUSTOM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['ruthlessRatio']),visualScope=3,step=3):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.distance = np.zeros((populationSize,populationSize),dtype=float)
        self.bestInd = copy.deepcopy(self.population.population[rd.randint(0, populationSize - 1)])
        self.worstInd = copy.deepcopy(self.population.population[rd.randint(0, populationSize - 1)])
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)

    def UpdateBestWorst(self):
        indexs = np.arange(self.population.populationSize)
        paretoFront = np.ones(self.population.populationSize)
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i], self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] += 1
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        worstIndex = np.argmax(paretoFront)
        self.bestInd = index
        self.worstInd =  worstIndex

    def CalculDistance(self):
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize) :
                dst = distance.euclidean((self.population.population[i]>0).astype(int), (self.population.population[j]>0).astype(int))
                self.distance[i,j] = dst

    def GetDominatingSolutions(self,i,isReverse):
        dominatingSolutions = []
        for j in range(self.population.populationSize):
            domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
            if isReverse:
                if domination == -1:
                    dominatingSolutions.append(j)
            else:
                if domination == 1:
                    dominatingSolutions.append(j)

        return dominatingSolutions

    def Turnament(self,participant,participantScore):
        bestIndex = 0
        bestScore = 0
        for p in range(len(participant)):
            if sum(participantScore[p])> bestScore:
                bestScore=sum(participantScore[p])
                bestIndex = p
        return bestIndex



    def UpdatePopulation(self,data):
        for i in range(self.population.populationSize):
            participants = []
            participantsScore = []
            dominatingSolutions = self.GetDominatingSolutions(i,False)
            isBetter = False
            if len(dominatingSolutions)>0:
                for j in dominatingSolutions:
                    nbChanges = int(self.distance[i][j]/2)
                    candidate = copy.deepcopy(self.population.population[i])
                    for k in range(nbChanges):
                        r = rd.random()
                        if r <0.33:
                            index = rd.choice(np.nonzero(self.population.population[j]>0))
                            candidate[index] += 1
                        elif r<0.66:
                            index = rd.choice(np.nonzero(self.population.population[i]>0))
                            candidate[index] += -1
                            index = rd.randint(0,self.nbItem*2-1)
                            candidate[index] += 1
                        else:
                            index = rd.randint(0, self.nbItem * 2 - 1)
                            candidate[index] += 1
                    is0 = self.population.CheckIfNullIndividual(candidate)
                    if type(is0) != bool:
                        candidate = copy.deepcopy(is0)
                    score = self.fitness.ComputeScoreIndividual(candidate,data)
                    domination = self.fitness.Domination(score,self.fitness.scores[i])
                    if domination == -1 :
                        participants.append(copy.deepcopy(candidate))
                        participantsScore.append(copy.deepcopy(score))
                        isBetter = True
                if isBetter == True:
                    indexBest = self.Turnament(participants,participantsScore)
                    self.population.population[i] = copy.deepcopy(participants[indexBest])
                    self.fitness.scores[i] = copy.deepcopy(participantsScore[indexBest])
                if isBetter == False:
                    self.population.population[i] = self.population.InitIndividual_HorizontalBinary()
                    self.fitness.scores[i] = self.fitness.ComputeScoreIndividual(self.population.population[i],data)
            else:
                participantsIndex = []
                dominatingSolutions = self.GetDominatingSolutions(i, True)
                for j in dominatingSolutions:
                    nbChanges = int(self.distance[i][j] / 2)
                    candidate = copy.deepcopy(self.population.population[j])
                    for k in range(nbChanges):
                        r = rd.random()
                        if r < 0.33:
                            index = rd.choice(np.nonzero(self.population.population[i] > 0))
                            candidate[index] += 1
                        elif r<0.66:
                            index = rd.choice(np.nonzero(self.population.population[j] > 0))
                            candidate[index] += -1
                            index = rd.randint(0, self.nbItem * 2 - 1)
                            candidate[index] += 1
                        else:
                            index = rd.randint(0, self.nbItem * 2 - 1)
                            candidate[index] += 1
                    is0 = self.population.CheckIfNullIndividual(candidate)
                    if type(is0) != bool:
                        candidate = copy.deepcopy(is0)
                    score = self.fitness.ComputeScoreIndividual(candidate,data)
                    domination = self.fitness.Domination(score, self.fitness.scores[j])
                    if domination == -1:
                        participantsIndex.append(j)
                        participants.append(copy.deepcopy(candidate))
                        participantsScore.append(copy.deepcopy(score))
                        isBetter = True
                if isBetter == True:
                    betterIndex = self.Turnament(participants, participantsScore)
                    self.population.population[participantsIndex[betterIndex]] = copy.deepcopy(participants[betterIndex])
                    self.fitness.scores[participantsIndex[betterIndex]] = copy.deepcopy(participantsScore[betterIndex])
                else:
                    self.population.population[i] = self.population.InitIndividual_HorizontalBinary()
                    self.fitness.scores[i] = self.fitness.ComputeScoreIndividual(self.population.population[i],data)




    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.fitness.paretoFront=np.zeros((1,len(self.objectivesNames)),dtype=float)
        self.fitness = Fitness('horizontal_binary', self.objectiveNames, self.population.populationSize)
        self.distance = np.zeros((self.population.populationSize, self.population.populationSize), dtype=float)
        self.bestInd = copy.deepcopy(self.population.population[rd.randint(0, self.population.populationSize - 1)])
        self.worstInd = copy.deepcopy(self.population.population[rd.randint(0, self.population.populationSize - 1)])
        self.executionTime = 0
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]
        self.fitness.ComputeScorePopulation(self.population.population, data)

    def Run(self,data,i):
        t1 = time()
        self.CalculDistance()
        self.UpdateBestWorst()
        self.UpdatePopulation(data)
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.executionTime = time() - t1



