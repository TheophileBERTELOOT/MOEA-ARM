import pandas as pd
import matplotlib.pyplot as plt
import copy

from src.Utils.Fitness import *
from src.Utils.Population import *
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *
"""
article :
"""



class MOPSO:
    def __init__(self,nbItem, populationSize, nbIteration, nbObjectifs, objectiveNames ,data,
                 hyperParameters = HyperParameters(['inertie', 'localAccelaration', 'globalAcceleration'])):
        self.population = Population('horizontal_binary',populationSize,nbItem)
        self.speeds = []
        self.personalBests = []
        self.personalBestsFitness = []
        self.globalBest = []
        self.globalBestFitness = []
        self.nbItem = nbItem
        self.inertie = hyperParameters.hyperParameters['inertie']
        self.localAcceleration = hyperParameters.hyperParameters['localAccelaration']
        self.globalAcceleration = hyperParameters.hyperParameters['globalAcceleration']
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.paretoFront = []
        self.fitness = Fitness('horizontal_binary',objectiveNames,self.population.populationSize)

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
        self.paretoFront = np.array([copy.deepcopy(self.globalBest)])
        self.globalBestFitness = np.array([[0 for i in range(self.nbObjectifs)]])

    def InitSpeed(self):
        self.speeds = []
        for i in range(self.population.populationSize):
            speed = []
            for j in range(self.nbItem*2):
                speed.append(0.0)
            self.speeds.append(speed)
        self.speeds = np.array(self.speeds)


    def UpdateParetoFront(self):
        bestIndexs = self.IdentifyPareto( self.fitness.scores)
        if len(bestIndexs) != 0:
            candidateParetoFront = copy.deepcopy(self.population.population[bestIndexs])
            candidateFitness = copy.deepcopy(self.fitness.scores[bestIndexs])
            population = np.concatenate([copy.deepcopy(self.paretoFront),candidateParetoFront],axis=0)
            populationScore = np.concatenate([copy.deepcopy(self.globalBestFitness),candidateFitness],axis=0)
            bestIndexs = self.IdentifyPareto( populationScore)
            self.paretoFront = population[bestIndexs]
            self.globalBestFitness = populationScore[bestIndexs]
            self.globalBest = rd.choice(self.paretoFront)

    def UpdatePersonalBest(self):
        for i in range(self.population.populationSize):
            if self.fitness.Domination(self.fitness.scores[i],self.personalBestsFitness[i]) == -1:
                self.personalBestsFitness[i] = self.fitness.scores[i]
                self.personalBests[i] = copy.deepcopy(self.population.population[i])

    def UpdateSpeed(self):
        for i in range(self.population.populationSize):
            r1 = rd.random()
            r2 = rd.random()
            self.speeds[i] = self.inertie*self.speeds[i]+self.localAcceleration*r1*(self.personalBests[i]-self.population.population[i])+self.globalAcceleration*r2*(self.globalBest-self.population.population[i])

    def IdentifyPareto(self,score):
        population_size = score.shape[0]
        population_ids = np.arange(population_size)
        pareto_front = np.ones(population_size, dtype=bool)
        uniqueScores = []
        for i in range(population_size):
            for j in range(population_size):
                domination = self.fitness.Domination(score[j],score[i])
                if domination == -1 :
                    pareto_front[i] = False
                    break
            if pareto_front[i] and not list(score[i]) in uniqueScores:
                uniqueScores.append(list(score[i]))
            else:
                pareto_front[i] = False

        return population_ids[pareto_front]

    def UpdatePosition(self):
        for i in range(self.population.populationSize):
            self.population.population[i] = self.population.population[i] + self.speeds[i]



    def Run(self,data,i):

        t1 = time()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdateParetoFront()
        self.UpdatePersonalBest()
        self.UpdateSpeed()
        self.UpdatePosition()
        self.executionTime = time() - t1





