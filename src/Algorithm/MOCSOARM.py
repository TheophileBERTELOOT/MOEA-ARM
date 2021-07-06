from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOCSOARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['ruthlessRatio']),visualScope=3,step=3):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.bestInd = copy.deepcopy(self.population.population[rd.randint(0,populationSize-1)])
        self.bestIndScore = 0
        self.visualScope = visualScope
        self.ruthlessRatio = hyperParameters.hyperParameters['ruthlessRatio']
        self.step = step
        self.distance = np.array([[0 for i in range(populationSize)] for j in range(populationSize)])
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)

    def FindLocalBest(self,matesScore):
        indexs = np.arange(len(matesScore))
        paretoFront = np.ones(len(matesScore))
        for i in range(len(matesScore)):
            for j in range(len(matesScore)):
                domination = self.fitness.Domination(matesScore[i], matesScore[j])
                if domination == 1:
                    paretoFront[i] = 0
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        return index

    def ChaseSwarming(self):
        step = rd.random()
        for i in range(self.population.populationSize):
            mates = []
            distance = self.distance[i,:]
            for j in range(self.population.populationSize):
                if distance[j] < self.visualScope:
                    mates.append(j)
            matesScore = self.fitness.scores[mates]
            localBest = self.FindLocalBest(matesScore)
            if localBest == i:
                self.population.population[i]+= step*(self.bestInd-self.population.population[i])
            else:
                self.population.population[i]+=step*(self.population.population[mates[localBest]]-self.population.population[i])

    def CalculDistance(self):
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize) :
                dst = distance.euclidean(self.population.population[i], self.population.population[j])
                self.distance[i,j] = dst

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

    def Dispersion(self):
        for i in range(self.population.populationSize):
            for j in range(self.step):
                index = rd.randint(0,self.nbItem*2-1)
                self.population.population[i][index] = float(rd.randint(-1,1))

    def RuthlessBehavior(self):
        for i in range(self.population.populationSize):
            r = rd.random()
            if r<self.ruthlessRatio:
                self.population.population[i] = copy.deepcopy(self.bestInd)

    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.ruthlessRatio = hyperParameters.hyperParameters['ruthlessRatio']
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]
        self.fitness.ComputeScorePopulation(self.population.population, data)

    def Run(self,data,i):
        t1 = time()
        self.CalculDistance()
        self.UpdateBestInd()
        self.ChaseSwarming()
        self.Dispersion()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBestInd()
        self.RuthlessBehavior()
        self.executionTime = time() - t1



