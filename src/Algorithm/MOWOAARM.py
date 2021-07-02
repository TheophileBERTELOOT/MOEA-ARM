from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOWOAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['b'])):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.bestInd = copy.deepcopy(self.population.population[rd.randint(0,populationSize-1)])
        self.bestIndScore = 0
        self.b = hyperParameters.hyperParameters['b']
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

    def ResetPopulation(self, data, hyperParameters):
        self.population.InitPopulation()
        self.b = hyperParameters.hyperParameters['b']
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.paretoFrontSolutions=[]
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBest()

    def Run(self,data,i):
        t1 = time()
        a = 2-2*(i/self.nbIteration)
        for i in range(self.population.populationSize):
            r = np.array([rd.random() for _ in range(self.nbItem*2)])
            A = 2 * a * r - a
            C = 2*r
            D = abs(C*self.bestInd-self.population.population[i])
            l = (rd.random()*2)-1
            p = rd.random()
            if p<0.5:
                if a < 1 :
                    self.population.population[i] = copy.deepcopy(self.bestInd)-A*D
                else:
                    index = rd.randint(0,self.population.populationSize-1)
                    D = abs(C*self.population.population[index]-self.population.population[i])
                    self.population.population[i] = copy.deepcopy(self.population.population[index]) -A*D
            else:
                self.population.population[i] = D*np.exp(self.b*l)*np.cos(2*np.pi*l)+self.bestInd
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBest()
        self.executionTime = time() - t1



