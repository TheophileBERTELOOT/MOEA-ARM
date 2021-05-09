import pandas as pd
import matplotlib.pyplot as plt
import copy

from src.Utils.Fitness import *
from src.Utils.Population import *
from src.Utils.Graphs import *
from time import time

"""
article :
"""



class MOPSO:
    def __init__(self,nbItem, populationSize, nbIteration, nbObjectifs, objectiveNames ,
                 inertie=0.5, localAccelaration = 0.5, globalAcceleration = 0.5,
                 save=True,display=True,path='Figures/'):
        self.population = Population('horizontal_binary',populationSize,nbItem)
        self.speeds = []
        self.personalBests = []
        self.personalBestsFitness = []
        self.globalBest = []
        self.globalBestFitness = []
        self.nbItem = nbItem
        self.inertie = inertie
        self.localAcceleration = localAccelaration
        self.globalAcceleration = globalAcceleration
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.paretoFront = []
        self.Fitness = Fitness('horizontal_binary',objectiveNames,self.population.populationSize)
        self.save = save
        self.display = display
        self.path = path
        self.executionTime = 0

        self.InitSpeed()
        self.InitPersonalBest()
        self.InitGlobalBest()



    def InitPersonalBest(self):
        self.personalBests = []
        for i in range(self.population.populationSize):
            personalBest = []
            self.personalBestsFitness.append([0.0 for _ in range(self.nbObjectifs)])
            for j in range(self.nbItem*2):
                personalBest.append(0.0)
            self.personalBests.append(personalBest)
        self.personalBests = np.array(self.personalBests)
        self.personalBestsFitness = np.array(self.personalBestsFitness)

    def InitGlobalBest(self):
        self.globalBest = []
        for i in range(self.nbItem*2):
            self.globalBest.append(0.0)
        self.globalBest = np.array(self.globalBest)

    def InitSpeed(self):
        self.speeds = []
        for i in range(self.population.populationSize):
            speed = []
            for j in range(self.nbItem*2):
                speed.append(0.0)
            self.speeds.append(speed)
        self.speeds = np.array(self.speeds)


    def UpdateParetoFront(self):
        bestIndexs = self.IdentifyPareto(self.population.population,self.Fitness.scores)
        if len(self.paretoFront) == 0:
            self.paretoFront = self.population.population[bestIndexs]
            self.globalBestFitness = self.Fitness.scores[bestIndexs]
        else:
            bestIndexs = self.IdentifyPareto(self.population.population, self.Fitness.scores)
            candidateParetoFront = self.population.population[bestIndexs]
            candidateFitness = self.Fitness.scores[bestIndexs]
            population = np.concatenate([self.paretoFront,candidateParetoFront],axis=0)
            populationScore = np.concatenate([self.globalBestFitness,candidateFitness],axis=0)
            bestIndexs = self.IdentifyPareto(population, populationScore)
            self.paretoFront = population[bestIndexs]
            self.globalBestFitness = populationScore[bestIndexs]

        self.globalBest = rd.choice(self.paretoFront)

    def UpdatePersonalBest(self):
        for i in range(self.population.populationSize):
            if self.Fitness.Domination(self.Fitness.scores[i],self.personalBestsFitness[i]) == -1:
                self.personalBestsFitness[i] = self.Fitness.scores[i]
                self.personalBests[i] = copy.deepcopy(self.population.population[i])

    def UpdateSpeed(self):
        for i in range(self.population.populationSize):
            r1 = rd.random()
            r2 = rd.random()
            self.speeds[i] = self.inertie*self.speeds[i]+self.localAcceleration*r1*(self.personalBests[i]-self.population.population[i])+self.globalAcceleration*r2*(self.globalBest-self.population.population[i])

    def IdentifyPareto(self,population,score):
        population_size = population.shape[0]
        population_ids = np.arange(population_size)
        pareto_front = np.ones(population_size, dtype=bool)
        for i in range(population_size):
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if all(score[j] >= score[i]) and any(score[j] > score[i]):
                    pareto_front[i] = 0
                    break
        return population_ids[pareto_front]

    def UpdatePosition(self):
        for i in range(self.population.populationSize):
            self.population.population[i] = self.population.population[i] + self.speeds[i]



    def Run(self,data,i):
        t1 = time()
        self.Fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdateParetoFront()
        self.UpdatePersonalBest()
        self.UpdateSpeed()
        self.UpdatePosition()
        self.executionTime = time() - t1
        print(self.globalBestFitness)




