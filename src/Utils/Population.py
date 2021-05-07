import numpy as np
import random as rd
import copy

class Population:
    def __init__(self,representation, populationSize,nbItem):
        self.representation = representation
        self.populationSize = populationSize
        self.nbItem = nbItem
        self.population  = []

        self.InitPopulation()

    def InitPopulation(self):
        self.population = []
        for i in range(self.populationSize):
            if self.representation == 'horizontal_binary':
                individual = self.InitIndividual_HorizontalBinary()
            if self.representation == 'horizontal_index':
                individual = self.InitIndividual_HorizontalIndex()
            self.population.append(individual)
        self.population = np.array(self.population)

    def SetPopulation(self,population):
        self.population = copy.deepcopy(population)

    def InitIndividual_HorizontalBinary(self):
        individual = []
        for i in range(self.nbItem):
            individual.append(-1.0)
        for i in range(self.nbItem):
            individual.append(float(rd.randint(-1,1)))
        for i in range(5):
            index = rd.randint(0,self.nbItem-1)
            individual[index] = 1.0
        return np.array(individual)

    def InitIndividual_HorizontalIndex(self):
        individual = []
        for i in range(self.nbItem):
            individual.append(0.0)
        for i in range(5):
            individual[i] = rd.randint(0,self.nbItem-1)
        return np.array(individual)

