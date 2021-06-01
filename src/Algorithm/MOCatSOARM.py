from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *


class MOCatSOARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                hyperParameters = HyperParameters(['mixtureRatio','velocityRatio']) ,velocitySize=3,SMP=10, SRD = 3):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize)
        self.mixtureRatio = hyperParameters.hyperParameters['mixtureRatio']
        self.velocityRatio = hyperParameters.hyperParameters['velocityRatio']
        self.velocitySize = velocitySize
        self.SMP = SMP
        self.SRD =SRD
        self.bestCat = np.zeros(nbItem*2,dtype=float)
        self.bestCatScore = np.zeros(nbObjectifs,dtype=float)

        self.executionTime = 0

    def GenerateVelocity(self):
        velocity = [0 for _ in range(self.nbItem*2)]
        nbChanges = rd.randint(1,self.velocitySize)
        for i in range(nbChanges):
            index = rd.randint(0,self.nbItem*2-1)
            velocity[index] = float(rd.randint(-1,1))
        return np.array(velocity)

    def UpdateBestInd(self):
        for i in range(self.population.populationSize):
            if self.fitness.Domination(self.fitness.scores[i], self.bestCatScore) == -1:
                self.bestCatScore = self.fitness.scores[i]
                self.bestCat = copy.deepcopy(self.population.population[i])

    def Resting(self,k,data):
        environment = []
        scores = []
        wheel = [i  for i in range(self.SMP)]
        for i in range(self.SMP):
            position = copy.deepcopy(self.population.population[k])
            for j in range(self.SRD):
                index = rd.randint(0,self.nbItem*2-1)
                position[index] = -1*position[index]
            score = self.fitness.ComputeScoreIndividual(position,data)
            environment.append(position)
            scores.append(score)

        for i in range(self.SMP):
            for j in range(self.SMP):
                if self.fitness.Domination(scores[i],scores[j]) == -1:
                    wheel.append(i)

        index = rd.choice(wheel)
        self.population.population[k] = environment[index]



    def Run(self,data,i):
        t1 = time()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdateBestInd()
        for j in range(self.population.populationSize):
            r = rd.random()
            if r<self.mixtureRatio:
                velocity = self.GenerateVelocity()
                r = rd.random()
                self.population.population[j] =self.population.population[j]+ velocity+r*self.velocityRatio*(self.bestCat-self.population.population[j])
            else:
                self.Resting(j,data)
        self.executionTime = time() - t1



