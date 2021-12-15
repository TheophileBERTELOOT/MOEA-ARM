from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOSOSARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['ruthlessRatio']),nbParasitismModification=5):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize,nbItem )
        self.bestInd = copy.deepcopy(self.population.population[rd.randint(0,populationSize-1)])
        self.bestIndScore = np.zeros(nbObjectifs,dtype=float)
        self.nbParasitismModification = nbParasitismModification
        self.distance = np.array([[0 for i in range(populationSize)] for j in range(populationSize)])
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBest()

    def UpdateBest(self):
        indexs = np.arange(self.population.populationSize)
        paretoFront = np.ones(self.population.populationSize)
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] = 0
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        self.bestInd = copy.deepcopy(self.population.population[index])
        self.bestIndScore = copy.deepcopy(self.fitness.scores[index])

    def Mutualism(self,data):
        for i in range(self.population.populationSize):
            j = rd.randint(0,self.population.populationSize-1)
            while j == i:
                j = rd.randint(0, self.population.populationSize - 1)
            mutualVector = (self.population.population[i]+self.population.population[j])/2
            BF1 = rd.randint(1,2)
            if BF1 ==1 :
                BF2 = 2
            else:
                BF2 = 1
            iNew = copy.deepcopy(self.population.population[i]) + rd.random() * (self.bestInd - mutualVector*BF1)
            iNewScore = self.fitness.ComputeScoreIndividual(iNew,data)
            jNew = copy.deepcopy(self.population.population[j]) + rd.random() * (self.bestInd - mutualVector*BF2)
            jNewScore = self.fitness.ComputeScoreIndividual(jNew, data)
            dominationI = self.fitness.Domination(self.fitness.scores[i],iNewScore)
            if dominationI == 1:
                self.population.population[i] = copy.deepcopy(iNew)
            dominationJ = self.fitness.Domination(self.fitness.scores[j],jNewScore)
            if dominationJ == 1:
                self.population.population[j] = copy.deepcopy(jNew)

    def Commensalism(self,data):
        for i in range(self.population.populationSize):
            j = rd.randint(0,self.population.populationSize-1)
            while j == i:
                j = rd.randint(0, self.population.populationSize - 1)
            iNew = self.population.population[i]+((rd.random()*2)-1)*(self.bestInd-self.population.population[j])
            iNewScore = self.fitness.ComputeScoreIndividual(iNew, data)
            dominationI = self.fitness.Domination(self.fitness.scores[i],iNewScore)
            if dominationI == 1:
                self.population.population[i] = copy.deepcopy(iNew)

    def Parasitism(self,data):
        for i in range(self.population.populationSize):
            j = rd.randint(0,self.population.populationSize-1)
            while j == i:
                j = rd.randint(0, self.population.populationSize - 1)
            iNew = copy.deepcopy(self.population.population[i])
            nbChange = rd.randint(1,self.nbParasitismModification)
            for k in range(nbChange):
                index = rd.randint(0,self.nbItem*2-1)
                iNew[index] = -1*iNew[index]
            iNewScore = self.fitness.ComputeScoreIndividual(iNew, data)
            dominationI = self.fitness.Domination(self.fitness.scores[j], iNewScore)
            if dominationI == 1:
                self.population.population[j] = copy.deepcopy(iNew)

    def Run(self,data,i):
        t1 = time()

        self.Mutualism(data)
        self.Commensalism(data)
        self.Parasitism(data)
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)


        self.executionTime = time() - t1



