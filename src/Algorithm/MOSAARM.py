from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOSAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 tempInitial = 1.0,nbIterationPerTemp=10,nbChanges=5,hyperParameters = HyperParameters(['alpha'])):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize,nbItem)
        self.temp = tempInitial
        self.tempInitial = tempInitial
        self.nbIterationPerTemp = nbIterationPerTemp
        self.nbChanges = nbChanges
        self.alpha = hyperParameters.hyperParameters['alpha']
        self.executionTime = 0

    def GenerateRule(self,i):
        ind = copy.deepcopy(self.population.population[i])
        nbChange = rd.randint(1,self.nbChanges)
        for j in range(nbChange):
            index = rd.randint(0,(self.nbItem*2)-1)
            ind[index]*=-1
        return ind

    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.alpha = hyperParameters.hyperParameters['alpha']
        self.temp = self.tempInitial
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]

    def Run(self,data,i):

        t1 = time()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        for j in range(self.population.populationSize):
            for m in range(self.nbIterationPerTemp):
                newRule = copy.deepcopy(self.GenerateRule(j))
                newRuleScore = copy.deepcopy(self.fitness.ComputeScoreIndividual(newRule,data))
                domination = self.fitness.Domination(self.fitness.scores[j],newRuleScore)
                delta =   abs(sum(newRuleScore)-sum(self.fitness.scores[j]))
                if domination == 1:
                    self.population.population[j] = copy.deepcopy(newRule)
                else:
                    r = rd.random()
                    if np.exp(-delta/self.temp)>r and delta != 0 and sum(newRuleScore) !=0:
                        self.population.population[j] = copy.deepcopy(newRule)
        self.temp*=self.alpha
        self.executionTime = time()-t1
