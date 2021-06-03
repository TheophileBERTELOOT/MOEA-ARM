from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOGEAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['Jr','Sr','epsilon'])):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.bestInd = np.zeros(nbItem*2,dtype=float)
        self.bestIndScore = np.zeros(nbObjectifs,dtype=float)
        self.worstInd = np.zeros(nbItem * 2, dtype=float)
        self.worstIndScore = np.zeros(nbObjectifs, dtype=float)
        self.mutatedVectors = np.zeros((populationSize, nbItem * 2), dtype=float)
        self.Jr = hyperParameters.hyperParameters['Jr']
        self.Sr = hyperParameters.hyperParameters['Sr']
        self.epsilon = hyperParameters.hyperParameters['epsilon']
        self.mutatedVectors = np.zeros((populationSize,nbItem*2),dtype=float)
        self.Si = np.ones(populationSize,dtype=float)
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)



    def UpdateBestWorst(self):
        indexs = np.arange(self.population.populationSize)
        paretoFront = np.ones(self.population.populationSize)
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] += 1
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        worstIndex = np.argmax(paretoFront)
        self.bestInd = copy.deepcopy(self.population.population[index])
        self.bestIndScore = copy.deepcopy(self.fitness.scores[index])
        self.worstInd = copy.deepcopy(self.population.population[worstIndex])
        self.worstIndScore = copy.deepcopy(self.fitness.scores[worstIndex])

    def VectorUpdating(self):
        for i in range(self.population.populationSize):
            rg = np.random.normal()
            ra = np.random.normal()
            dx = (abs(self.population.population[i]-self.bestInd)+abs(self.worstInd-self.population.population[i]))/2
            gmove = (rg*dx/2)*((self.worstInd-self.bestInd)/(self.worstInd-self.population.population[i]+self.bestInd+0.00001))
            gmove = self.population.CheckDivide0(gmove)
            acc = ra*(self.bestInd-self.population.population[i])
            self.mutatedVectors[i] = self.population.population[i]- gmove + acc

    def VectorJumping(self):
        for i in range(self.population.populationSize):
            r = np.random.normal()
            if r <self.Jr:
                rm = rd.random()
                index = rd.randint(0,self.population.populationSize-1)
                individual = self.mutatedVectors[i] +rm*(self.mutatedVectors[i] - self.mutatedVectors[index])
                individual = self.population.CheckDivide0(individual)
                is0 = self.population.CheckIfNullIndividual(individual)
                if type(is0) != bool:
                    individual = copy.deepcopy(is0)
                self.mutatedVectors[i] = copy.deepcopy(individual)

    def VectorRefreshing(self,data):
        for i in range(self.population.populationSize):
            score = self.fitness.ComputeScoreIndividual(self.mutatedVectors[i],data)
            domination = self.fitness.Domination(self.fitness.scores[i],score)
            if domination == 1:
                self.population.population[i] = copy.deepcopy(self.mutatedVectors[i])
            else:
                self.Si[i] = self.Si[i]-self.epsilon*self.Si[i]
                if self.Si[i]<self.Sr:
                    self.population.population[i] = self.population.InitIndividual_HorizontalBinary()
                    self.Si[i] = 1

    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.Jr = hyperParameters.hyperParameters['Jr']
        self.Sr = hyperParameters.hyperParameters['Sr']
        self.epsilon = hyperParameters.hyperParameters['epsilon']
        self.fitness.ComputeScorePopulation(self.population.population, data)

    def Run(self,data,i):

        t1 = time()
        self.UpdateBestWorst()
        self.VectorUpdating()
        self.VectorJumping()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.VectorRefreshing(data)
        self.executionTime = time() - t1



