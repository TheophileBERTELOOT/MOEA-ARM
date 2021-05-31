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

    def GetIndividualRepresentation(self,individual):
        if self.representation == 'horizontal_binary':
            presence = individual[:int(len(individual) / 2)]
            location = individual[int(len(individual) / 2):]
            indexRule = (presence > 0).nonzero()[0]
            indexAntecedent = indexRule[(location[indexRule] < 0).nonzero()[0]]
            indexConsequent = indexRule[(location[indexRule] > 0).nonzero()[0]]
        elif self.representation == 'horizontal_index':
            individual = individual[0]
            indexRule = (individual[1:] > 0).nonzero()[0]
            indexAntecedent = (individual[1:int(individual[0])] > 0).nonzero()[0]
            indexConsequent = (individual[int(individual[0]):] > 0).nonzero()[0]
        return indexRule,indexAntecedent,indexConsequent

    def CheckIfNull(self):
        for i in range(self.populationSize):
            indexRule,indexAntecedent,indexConsequent = self.GetIndividualRepresentation(self.population[i])
            if len(indexAntecedent) <1 or len(indexConsequent<1):
                if self.representation == 'horizontal_binary':
                    individual = self.InitIndividual_HorizontalBinary()
                if self.representation == 'horizontal_index':
                    individual = self.InitIndividual_HorizontalIndex()
                self.population[i] = copy.deepcopy(individual)

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

