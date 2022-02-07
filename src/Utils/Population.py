import numpy as np
import random as rd
import copy




class Population:
    def __init__(self,representation, populationSize,nbItem,isPolypharmacy = False):
        self.representation = representation
        self.populationSize = populationSize
        self.nbItem = nbItem
        self.population  = []
        self.isPolypharmacy = isPolypharmacy
        self.minLength = 1
        if isPolypharmacy:
            self.minLength = 5
        self.InitPopulation()

    def GetIndividualRepresentation(self,individual):
        if self.representation == 'horizontal_binary':
            presence = individual[:int(len(individual) / 2)]
            location = individual[int(len(individual) / 2):]
            indexRule = (presence > 0).nonzero()[0]
            indexAntecedent = indexRule[(location[indexRule] <= 0).nonzero()[0]]
            indexConsequent = indexRule[(location[indexRule] > 0).nonzero()[0]]
        elif self.representation == 'horizontal_index':
            individual = individual[0]
            indexRule = (individual[1:] >= 0).nonzero()[0]
            indexAntecedent = (individual[1:int(individual[0])] > 0).nonzero()[0]
            indexConsequent = (individual[int(individual[0]):] > 0).nonzero()[0]
        return indexRule,indexAntecedent,indexConsequent

    def GetPopulationRepresentation(self,isString):
        popRep = []
        for i in range(self.populationSize):
            indexRule,indexAntecedent,indexConsequent = self.GetIndividualRepresentation(self.population[i])
            if isString:
                popRep.append([str(indexRule),str(indexAntecedent),str(indexConsequent)])
            else:
                popRep.append([list(indexRule), list(indexAntecedent), list(indexConsequent)])
        return np.array(popRep)



    def AddIndividualToTested(self,individual):
        rep = self.GetIndividualRepresentation(individual)
        tsneFriendlyRep = np.zeros(self.nbItem*2)
        for antecedent in rep[1]:
            tsneFriendlyRep[antecedent]=1
        for consequent in rep[2]:
            tsneFriendlyRep[self.nbItem+consequent] =1
        return tsneFriendlyRep

    def CheckIfNullIndividual(self,ind):
        indexRule,indexAntecedent,indexConsequent = self.GetIndividualRepresentation(ind)
        if len(indexAntecedent) <self.minLength  or len(indexConsequent)<self.minLength :
            if self.representation == 'horizontal_binary':
                individual = self.InitIndividual_HorizontalBinary()
            if self.representation == 'horizontal_index':
                individual = self.InitIndividual_HorizontalIndex()
            return copy.deepcopy(individual)
        return False

    def CheckIfNull(self):
        for i in range(self.populationSize):
            indexRule,indexAntecedent,indexConsequent = self.GetIndividualRepresentation(self.population[i])
            while len(indexAntecedent) <self.minLength  or len(indexConsequent)<self.minLength :
                if self.representation == 'horizontal_binary':
                    individual = self.InitIndividual_HorizontalBinary()
                if self.representation == 'horizontal_index':
                    individual = self.InitIndividual_HorizontalIndex()
                self.population[i] = copy.deepcopy(individual)
                indexRule, indexAntecedent, indexConsequent = self.GetIndividualRepresentation(self.population[i])


    def InitPopulation(self):
        self.population = []
        for i in range(self.populationSize):
            if self.representation == 'horizontal_binary':
                individual = self.InitIndividual_HorizontalBinary()
            if self.representation == 'horizontal_index':
                individual = self.InitIndividual_HorizontalIndex()
            self.population.append(individual)
        self.population = np.array(self.population)
        self.CheckIfNull()
        # g = Graphs([],self.testedPopulation)
        # g.dataTSNE()

    def SetPopulation(self,population):
        self.population = copy.deepcopy(population)

    def InitIndividual_HorizontalBinary(self,isPolypharmacy=False):
        individual = []
        nbDefaultItem = 2
        if isPolypharmacy:
            nbDefaultItem = 5
        for i in range(self.nbItem):
            individual.append(-1.0)
        for i in range(self.nbItem):
            individual.append(float(0.0))
        for i in range(nbDefaultItem):
            index = rd.randint(0,self.nbItem-1)
            individual[index] = 1.0
            if i == 0:
                individual[self.nbItem+index] = 1.0
            else:
                individual[self.nbItem + index] = -1.0
        return np.array(individual)

    def CheckDivide0(self,p):
        p = np.where(p==np.inf,0,p)
        p =np.where(p==-np.inf,0,p)
        p = np.where(np.isnan(p),0,p)
        return p

    def InitIndividual_HorizontalIndex(self):
        individual = []
        for i in range(self.nbItem):
            individual.append(0.0)
        for i in range(5):
            individual[i] = rd.randint(0,self.nbItem-1)
        return np.array(individual)



