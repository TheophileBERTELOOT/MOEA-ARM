from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOGSAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['G'])):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.bestInd = np.zeros(nbItem * 2, dtype=float)
        self.bestIndScore = np.zeros(nbObjectifs, dtype=float)
        self.worstInd = np.zeros(nbItem * 2, dtype=float)
        self.worstIndScore = np.zeros(nbObjectifs, dtype=float)
        self.masses = np.zeros(populationSize, dtype=float)
        self.distance=np.zeros((populationSize,populationSize),dtype=float)
        self.velocity = np.zeros((populationSize,nbItem*2),dtype=float)
        self.forces = np.zeros((populationSize,nbItem*2), dtype=float)
        self.G= hyperParameters.hyperParameters['G']
        self.epsilon = 0.0000001
        self.executionTime = 0

    def UpdateG(self):
        self.G = self.G*self.G

    def UpdateMasse(self):
        self.masses = np.zeros(self.population.populationSize, dtype=float)
        for i in range(self.population.populationSize):
            self.masses[i] = (sum(self.fitness.scores[i])-sum(self.worstIndScore))/(sum(self.bestIndScore)-sum(self.worstIndScore))
            self.masses[i] = self.population.CheckDivide0(self.masses[i])
        somme = sum(self.masses)
        self.masses=self.masses/somme
        self.masses = self.population.CheckDivide0(self.masses)

    def CalculDistance(self):
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                dst = distance.euclidean(self.population.population[i], self.population.population[j])
                self.distance[i, j] = dst

    def UpdateAcceleration(self):
        for i in range(self.population.populationSize):
            somme = np.zeros(self.nbItem*2,dtype=float)
            for j in range(self.population.populationSize):
                somme+= 2*((self.masses[i]*self.masses[j])/(self.distance[i,j]+self.epsilon))*(self.population.population[j]-self.population.population[i])
            self.forces[i] = copy.deepcopy(somme)

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

    def UpdatePosition(self,data):
        for i in range(self.population.populationSize):
            self.velocity[i] = self.velocity[i] *rd.random()+ self.forces[i]/self.masses[i]
            self.velocity[i] = self.population.CheckDivide0(self.velocity[i])
            individual = copy.deepcopy(self.population.population[i] ) + self.velocity[i]
            isNull = self.population.CheckIfNullIndividual(individual)
            if type(isNull)!=bool:
                individual = self.population.InitIndividual_HorizontalBinary()
            score = self.fitness.ComputeScoreIndividual(individual,data)
            domination = self.fitness.Domination(self.fitness.scores[i],score)
            if domination == 1:
                self.population.population[i] = copy.deepcopy(individual)


    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.G = hyperParameters.hyperParameters['G']
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.paretoFrontSolutions=[]


    def Run(self,data,i):
        t1 = time()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBestWorst()
        self.UpdateG()
        self.UpdateMasse()
        self.CalculDistance()
        self.UpdateAcceleration()
        self.UpdatePosition(data)
        self.executionTime = time() - t1



