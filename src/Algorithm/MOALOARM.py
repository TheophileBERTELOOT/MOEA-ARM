from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time

class MOALOARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 nbChanges = 5,
                 save=True,display=True,path='Figures/'):
        self.ants = Population('horizontal_binary', int(populationSize/2), nbItem)
        self.lions = Population('horizontal_binary', int(populationSize/2), nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, int(populationSize/2) )
        self.fitnessAnts = Fitness('horizontal_binary', objectiveNames, int(populationSize/2) )
        self.elite = []
        self.eliteScore = []
        self.maxBound = 1
        self.minBound = -1
        self.nbChanges = nbChanges
        self.save = save
        self.display = display
        self.path = path
        self.executionTime = 0
        self.fitnessAnts.ComputeScorePopulation(self.ants.population, data)
        self.fitness.ComputeScorePopulation(self.lions.population, data)
        self.UpdateElite()



    def UpdateElite(self):
        indexs = np.arange(self.lions.populationSize)
        paretoFront = np.ones(self.lions.populationSize)
        for i in range(self.lions.populationSize):
            for j in range(self.lions.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] = 0
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        self.elite = copy.deepcopy(self.lions.population[index])
        self.eliteScore = copy.deepcopy(self.fitness.scores[index])

    def SelectLions(self):

        candidate = []
        for i in range(self.lions.populationSize):
            for j in range(self.lions.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
                if domination == 1:
                    candidate.append(i)
        index = rd.choice(candidate)
        return index

    def UpdateBounds(self,i):
        i_ = i+1
        if i_ >0.95*self.nbIteration:
            w = 6
        elif i_ >0.9*self.nbIteration:
            w = 5
        elif i_ > 0.75 * self.nbIteration:
            w = 4
        elif i_ >0.5*self.nbIteration:
            w = 3
        else:
            w = 2

        self.maxBound = self.maxBound/(np.power(10,w)*(i_/self.nbIteration))
        self.minBound = self.minBound / (np.power(10, w) * (i_ / self.nbIteration))

    def GenerateRandomWalk(self):
        rdw = [0 for _ in range(self.nbItem*2)]
        nbChanges = rd.randint(1,self.nbChanges)
        for i in range(nbChanges):
            index = rd.randint(0,self.nbItem*2-1)
            rdw[index] = float(rd.randint(-1,1))
        rdw = np.array(rdw)
        return rdw


    def Run(self,data,i):
        t1 = time()
        for i in range(self.ants.populationSize):
            lion = self.SelectLions()
            self.UpdateBounds(i)
            lionRandomWalk = self.lions.population[lion] + self.GenerateRandomWalk()
            eliteRandomWalk = copy.deepcopy(self.elite) + self.GenerateRandomWalk()
            self.ants.population[i] = (lionRandomWalk+eliteRandomWalk)/2
            score = self.fitnessAnts.ComputeScoreIndividual(self.ants.population[i],data)
            domination = self.fitnessAnts.Domination(self.fitness.scores[lion],score)
            self.fitnessAnts.scores[i] = copy.deepcopy(score)
            if domination == 1:
                self.lions.population[lion] = copy.deepcopy(self.ants.population[i])
        self.fitness.ComputeScorePopulation(self.lions.population,data)
        self.UpdateElite()
        self.executionTime = time() - t1



