from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MODEARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['F','CR'])):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.mutatedVectors = np.zeros((populationSize,nbItem*2),dtype=float)
        self.F = hyperParameters.hyperParameters['F']
        self.CR=hyperParameters.hyperParameters['CR']
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)


    def Selection(self,data):
        for i in range(self.population.populationSize):
            score = self.fitness.ComputeScoreIndividual(self.mutatedVectors[i],data)
            domination = self.fitness.Domination(self.fitness.scores[i],score)
            if domination == 1:
                self.population.population[i] = copy.deepcopy(self.mutatedVectors[i])
            if domination == 0:
                if sum(self.fitness.scores[i])<sum(score):
                    self.population.population[i] = copy.deepcopy(self.mutatedVectors[i])


    def CrossOver(self):
        for i in range(self.population.populationSize):
            q = rd.randint(0,self.nbItem*2-1)
            for j in range(self.nbItem*2):
                r = rd.random()
                if r>self.CR and j != q:
                    self.mutatedVectors[i][j] = self.population.population[i][j]

    def Mutation(self):
        for i in range(self.population.populationSize):
            a = rd.randint(0,self.population.populationSize-1)
            b = rd.randint(0,self.population.populationSize-1)
            c = rd.randint(0,self.population.populationSize-1)
            while a==i or b==i or c==i or a==b or a==c or b==c:
                a = rd.randint(0, self.population.populationSize - 1)
                b = rd.randint(0, self.population.populationSize - 1)
                c = rd.randint(0, self.population.populationSize - 1)
            self.mutatedVectors[i] = self.population.population[a] + self.F*(self.population.population[b]-self.population.population[c])


    def Run(self,data,i):

        t1 = time()
        self.Mutation()
        self.CrossOver()
        self.Selection(data)
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.executionTime = time() - t1



