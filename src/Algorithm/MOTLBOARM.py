from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
import numpy as np

class MOTLBOARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.teacher = []
        self.TF = 0
        self.executionTime = 0

    def UpdateTeacher(self):
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
        self.teacher = copy.deepcopy(self.population.population[index])


    def Run(self,data,i):
        t1 = time()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        M = np.array([np.average(self.population.population[:,i]) for i in range(self.nbItem*2)])
        self.UpdateTeacher()
        self.TF = np.round(1+rd.random()*rd.randint(1,2))
        self.Diff = rd.random()*(self.teacher-self.TF*M)
        for i in range(self.population.populationSize):
            xijp = self.population.population[i] + self.Diff
            score = self.fitness.ComputeScoreIndividual(xijp,data)
            domination =self.fitness.Domination(score,self.fitness.scores[i])
            if domination == -1:
                self.population.population[i] = copy.deepcopy(xijp)
            a = rd.randint(0,self.population.populationSize-1)
            b = rd.randint(0, self.population.populationSize-1)
            domination = self.fitness.Domination(self.fitness.scores[a],self.fitness.scores[b])
            if domination == -1 :
                xjaipp = self.population.population[a] + rd.random() *(self.population.population[a] - self.population.population[b])
            else:
                xjaipp = self.population.population[a] + rd.random() * (
                            self.population.population[b] - self.population.population[a])
            score = self.fitness.ComputeScoreIndividual(xjaipp,data)
            domination = self.fitness.Domination(self.fitness.scores[a],score)
            if domination == 1:
                self.population.population[a] = copy.deepcopy(xjaipp)
        self.executionTime = time() - t1