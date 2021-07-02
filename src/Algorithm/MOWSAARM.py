from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class MOWSAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 visionRange=3,nbPrey=10,step=5,hyperParameters = HyperParameters(['velocityFactor','enemyProb'])):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize)
        self.visionRange = visionRange
        self.step = step
        self.velocityFactor = hyperParameters.hyperParameters['velocityFactor']
        self.enemyProb = hyperParameters.hyperParameters['enemyProb']
        self.nbPrey = nbPrey

        self.executionTime = 0

    def GenerateRule(self, i,data):
        for k in range(self.nbPrey):
            nbChanges = rd.randint(1,self.step)
            ind = copy.deepcopy(self.population.population[i])
            for j in range(nbChanges):
                index = rd.randint(0, (self.nbItem * 2) - 1)
                ind[index] *= -1
            score = self.fitness.ComputeScoreIndividual(ind,data)
            if k == 0:
                bestPrey=ind
                bestPreyScore=score
            else:
                domination = self.fitness.Domination(bestPreyScore,score)
                if domination == 1 :
                    bestPreyScore = score
                    bestPrey = ind

        return bestPrey,bestPreyScore

    def CalculDistance(self,wolf,prey):
        return distance.euclidean(wolf,prey)

    def GenerateEscape(self):
        return self.population.InitIndividual_HorizontalBinary()

    def ResetPopulation(self, data, hyperParameters):
        self.population.InitPopulation()
        self.velocityFactor = hyperParameters.hyperParameters['velocityFactor']
        self.enemyProb = hyperParameters.hyperParameters['enemyProb']
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.paretoFrontSolutions=[]


    def Run(self,data,i):

        t1 = time()
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        for j in range(self.population.populationSize):
            prey,preyScore = self.GenerateRule(j,data)
            dist = self.CalculDistance(self.population.population[j],prey)
            domination = self.fitness.Domination(self.fitness.scores[j],preyScore)
            if dist < self.visionRange and domination == 1:
                beta0 = sum(preyScore)
                escape =  (rd.random()-0.5)/100
                self.population.population[j] = self.population.population[j]+beta0*np.exp(-dist**2)*(prey-self.population.population[j])+escape
            else:
                r = rd.random()-0.5
                self.population.population[j] = self.population.population[j]+self.velocityFactor*r
            r = rd.random()
            if r<self.enemyProb:
                escapeSpot = self.GenerateEscape()
                self.population.population[j] = escapeSpot
        self.executionTime = time() - t1

