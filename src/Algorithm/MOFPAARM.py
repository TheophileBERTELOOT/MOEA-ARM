from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
import numpy as np
from scipy.stats import levy
from src.Utils.HyperParameters import *

class MOFPAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                hyperParameters = HyperParameters(['P','gamma']) ,lambd= 1.5 , nbChanges=5):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.P = hyperParameters.hyperParameters['P']
        self.lambd = lambd
        self.bestSolution = []
        self.bestSolutionScore = 0
        self.gamma = hyperParameters.hyperParameters['gamma']
        self.nbChanges =nbChanges
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBestSolution()

    def Levy(self):
        return levy.rvs(loc=-1,scale=0.5,size=self.nbItem*2)



    def UpdateBestSolution(self):
        indexs = np.arange(self.population.populationSize)
        paretoFront = np.ones(self.population.populationSize)
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i], self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] = 0
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        self.bestSolution = copy.deepcopy(self.population.population[index])

    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.P = hyperParameters.hyperParameters['P']
        self.gamma = hyperParameters.hyperParameters['gamma']
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBestSolution()

    def Run(self,data,i):

        t1 = time()
        r = rd.random()
        if r>self.P:
            for i in range(self.population.populationSize):
                n = self.population.population[i] +self.gamma * self.Levy()*(self.bestSolution-self.population.population[i])
                score = self.fitness.ComputeScoreIndividual(n,data)
                domination = self.fitness.Domination(score, self.fitness.scores[i])
                if domination == -1:
                    self.population.population[i] = copy.deepcopy(n)
        else:
            for i in range(self.population.populationSize):
                pol1 = self.population.population[rd.randint(0,self.population.populationSize-1)]
                pol2 = self.population.population[rd.randint(0, self.population.populationSize - 1)]
                n = self.population.population[i] + rd.random()*(pol1-pol2)
                score = self.fitness.ComputeScoreIndividual(n, data)
                domination = self.fitness.Domination(score, self.fitness.scores[i])
                if domination == -1:
                    self.population.population[i] = copy.deepcopy(n)
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdateBestSolution()
        self.executionTime = time() - t1