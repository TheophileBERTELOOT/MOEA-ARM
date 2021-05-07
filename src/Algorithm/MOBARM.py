import matplotlib.pyplot as plt
import copy

from src.Utils.Fitness import *
from src.Utils.Population import *
from src.Utils.Graphs import *
"""
article :
"""

class MOBARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbPoints, nbObjectifs, objectiveNames ,
                 beta = 0.7,gamma = 0.1, alpha =0.9,
                 save=True,display=True,path='Figures/'):
        self.population = Population('horizontal_index',populationSize,nbItem)
        self.fitnessScore = []
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.nbPoints = nbPoints
        self.frontPareto = []
        self.frontParetoFitness = []
        self.coefs = []
        self.initialPulseRate =rd.random()
        self.pulseRate = self.initialPulseRate
        self.loudness = rd.random()
        self.fitness = Fitness('horizontal_index', objectiveNames, self.population.populationSize)
        self.bestFitness = 0
        self.bestIndividual = np.array([])
        self.save = save
        self.display = display
        self.path = path

        self.InitCoefs()
        self.InitPopulation()

    def InitIndividual(self):
        frequency = rd.randint(0,self.nbItem-1)
        velocity = rd.randint(0,self.nbItem)
        individual = [frequency,velocity,3]
        return individual


    def InitCoefs(self):
        coefs = np.array([rd.random() for _ in range(self.nbObjectifs)])
        coefs/=np.sum(coefs)
        self.coefs = coefs

    def InitPopulation(self):
        self.pulseRate = self.initialPulseRate
        self.loudness = rd.random()
        bats = []
        for i in range(self.population.populationSize):
            individual = self.InitIndividual()
            bats.append(individual)
        bats = np.array(bats)
        bats = np.concatenate([bats,self.population.population],axis=1)
        self.population.SetPopulation(bats)

    def UpdateIndividual(self,individual):
        frequency = int(individual[0])
        velocity = int(individual[1])
        velocityInitial = copy.deepcopy(velocity)
        law = individual[2:]
        while velocity< frequency:
            if rd.random()>self.loudness:
                law[velocity]+=1
            else:
                law[velocity]-=1
            if sum(np.isin(law,law[velocity]))>1:
                law[velocity] = 0
            if law[velocity]>self.nbItem or law[velocity]<0:
                law[velocity] = 0
            if law[0]==0:
                law[0] = 2
            if sum(law[1:int(law[0])]) == 0:
                law[1]=rd.randint(1,self.nbItem-1)
            if sum(law[int(law[0]):]) == 0:
                law[self.nbItem-1]=rd.randint(1,self.nbItem-1)
            velocity+=1
        individual[0] = 1 + self.nbItem*self.beta
        individual[1] = self.nbItem - individual[0] - velocityInitial


    def UpdatePopulation(self):
        for i in range(self.population.populationSize):
            self.UpdateIndividual(self.population.population[i])

    def UpdateBestIndividual(self):
        for i in range(self.population.populationSize):
            score = self.fitness.scores[i].dot(self.coefs)
            if self.bestFitness <= score:
                self.bestFitness = score
                self.bestIndividual = copy.deepcopy(self.population.population[i])

    def GetLocalSearch(self,data,t):
        candidat = copy.deepcopy(self.bestIndividual)
        law = candidat[2:]
        index = rd.randint(0,self.nbItem-1)
        value = rd.randint(0, self.nbItem - 1)
        while sum(np.isin(law,value))>1:
            value = rd.randint(0, self.nbItem - 1)
        candidat[2+index] = value


        score = self.fitness.ComputeScoreIndividual([law],data)
        score = score.dot(self.coefs)
        if score> self.bestFitness:
            self.bestFitness = score
            self.bestIndividual = candidat
            self.pulseRate = self.initialPulseRate*(1-np.exp(-self.gamma*t))
            self.loudness = self.alpha*self.loudness


    def Run(self,data):
        for i in range(self.nbPoints):
            t = 0
            while t < self.nbIteration:
                laws = self.population.population[:,[np.arange(2,self.nbItem)]]
                self.fitness.ComputeScorePopulation(laws,data)
                self.UpdatePopulation()
                self.UpdateBestIndividual()
                if rd.random()>self.pulseRate:
                    self.GetLocalSearch(data,t)
                t+=1
            self.frontPareto.append(self.bestIndividual)
            law = self.bestIndividual[2:]
            self.frontParetoFitness.append(self.fitness.ComputeScoreIndividual([law],data))
            print(np.array(self.frontParetoFitness))
            self.population.InitPopulation()
            self.InitPopulation()
            self.bestFitness = 0
            graph = Graphs(self.fitness.objectivesNames, self.fitness.scores, self.save, self.display,
                           self.path + str(i))
            graph.Graph3D()



