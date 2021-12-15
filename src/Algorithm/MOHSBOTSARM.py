from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOHSBOTSARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                    nbChanges = 5, nbNeighbours = 10,
                 ):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize,nbItem )
        self.bestInd = copy.deepcopy(self.population.population[rd.randint(0,populationSize-1)])
        self.bestIndScore = []
        self.tabooList = []
        self.danceTable = []
        self.danceTableScore = []

        self.executionTime = 0
        self.nbNeighbours = nbNeighbours
        self.nbChanges = nbChanges
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBestInd()


    def UpdateBestInd(self):
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

    def FindSearchRegion(self):
        for i in range(self.population.populationSize):
            index = rd.randint(0,self.nbItem*2-1)
            self.population.population[i] = self.population.InitIndividual_HorizontalBinary()
            self.population.population[i][index] = self.bestInd[index]

    def LocalSearch(self,bee,data):
        bestInd = []
        bestScore = np.zeros(self.nbObjectifs,dtype=float)
        for j in range(self.nbNeighbours):
            ind = copy.deepcopy(self.population.population[bee])
            nbChange = rd.randint(1,self.nbChanges)
            for i in range(nbChange):
                index = rd.randint(0,self.nbItem*2-1)
                ind[index] = -1*ind[index]
            score = self.fitness.ComputeScoreIndividual(ind,data)
            domination = self.fitness.Domination(bestScore,score)
            if domination == 1:
                bestScore = score
                bestInd = ind
        return bestInd,bestScore

    def BestOfDance(self):
        indexs = np.arange(self.population.populationSize)
        paretoFront = np.ones(self.population.populationSize)
        for i in range(self.population.populationSize):
            if len(self.danceTable[i]) != 0 :
                for j in range(self.population.populationSize):
                    domination = self.fitness.Domination(self.danceTableScore[i], self.danceTableScore[j])
                    if domination == 1:
                        paretoFront[i] = 0
                        break
            else:
                paretoFront[i] = 0

        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        self.bestInd = copy.deepcopy(self.danceTable[index])
        self.bestIndScore = copy.deepcopy(self.danceTableScore[index])

    def Run(self,data,i):

        t1 = time()
        self.danceTable = []
        self.danceTableScore = []
        self.tabooList.append(list(self.bestInd))
        self.FindSearchRegion()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        for j in range(self.population.populationSize):
            bestInd,bestScore = self.LocalSearch(j,data)
            self.danceTable.append(bestInd)
            self.danceTableScore.append(bestScore)
        self.BestOfDance()
        if list(self.bestInd) in self.tabooList:
            index = rd.randint(0,len(self.tabooList)-1)
            self.bestInd = copy.deepcopy(self.tabooList[index])
        self.executionTime = time() - t1



