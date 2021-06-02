from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOCMSARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['ruthlessRatio']),visualScope=10,step=3):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.force = np.zeros((populationSize,nbItem*2),dtype=float)
        self.velocity = np.zeros((populationSize, nbItem * 2), dtype=float)
        self.bestInd = copy.deepcopy(self.population.population[rd.randint(0,populationSize-1)])
        self.bestIndScore = np.zeros(nbObjectifs,dtype=float)
        self.worstInd = copy.deepcopy(self.population.population[rd.randint(0, populationSize - 1)])
        self.worstIndScore = np.zeros(nbObjectifs, dtype=float)

        self.ruthlessRatio = hyperParameters.hyperParameters['ruthlessRatio']

        self.distance = np.zeros((populationSize,populationSize),dtype=float)
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)


    def UpdateBestWorst(self):
        indexs = np.arange(self.population.populationSize)
        paretoFront = np.ones(self.population.populationSize)
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i], self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] += 1
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        worstIndex = np.argmax(paretoFront)
        self.bestInd = copy.deepcopy(self.population.population[index])
        self.bestIndScore = copy.deepcopy(self.fitness.scores[index])
        self.worstInd =  copy.deepcopy(self.population.population[worstIndex])
        self.worstIndScore = copy.deepcopy(self.fitness.scores[worstIndex])


    def CalculDistance(self):
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                dstij = distance.euclidean(self.population.population[i], self.population.population[j])
                distMB = distance.euclidean((self.population.population[i]+self.population.population[j])/2, self.bestInd)
                self.distance[i][j] = dstij/(distMB+0.000000000000000000000001)

    def CalculForce(self):
        a=0.1*self.nbItem*2
        for i in range(self.population.populationSize):
            somme = np.zeros(self.nbItem*2,dtype=float)
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
                if domination == 1:
                    pij = 1
                else:
                    pij = 0
                if self.distance[i][j]<a:
                    i1=1
                    i2=0
                else:
                    i1=0
                    i2=1
                if i != j:
                    somme = somme + (((sum(self.fitness.scores[j])*self.distance[i][j]*i1)/pow(a,3))+(sum(self.fitness.scores[j])*i2)/pow(self.distance[i][j],2))*pij*(self.population.population[j]-self.population.population[i])
            somme = sum(self.fitness.scores[i])*somme
            self.force[i] = somme

    def UpdatePosition(self,i,data):
        kv = 0.5*(1-i/self.nbIteration)
        ka = 0.5*(1+i/self.nbIteration)
        for i in range(self.population.populationSize):
            individual = rd.random()*ka*(self.force[i]/sum(self.fitness.scores[i]))+rd.random()*kv*self.velocity[i]+self.population.population[i]
            vNew = individual - self.population.population[i]
            score = self.fitness.ComputeScoreIndividual(individual,data)
            if self.fitness.Domination(self.worstIndScore,score) == 1:
                self.population.population[i] = copy.deepcopy(individual)
                self.velocity[i] = copy.deepcopy(vNew)
                self.UpdateBestWorst()

    def Run(self,data,i):

        t1 = time()
        self.UpdateBestWorst()
        self.CalculDistance()
        self.CalculForce()
        self.UpdatePosition(i,data)
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)

        self.executionTime = time() - t1



