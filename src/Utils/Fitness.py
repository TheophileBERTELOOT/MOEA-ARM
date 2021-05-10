import random as rd
import numpy as np
from numba import jit, cuda
from numba import njit

@njit
def SuppGPU( individual, data):
    supp = 0
    for i in range(data.shape[0]):
        if np.sum(data[i][individual]) == len(individual):
            supp += 1
    return supp / data.shape[0]

def negativeSuppGPU( individual, data):
    supp = 0
    for i in range(data.shape[0]):
        if np.sum(data[i][individual]) != len(individual):
            supp += 1
    return supp / data.shape[0]

class Fitness:
    def __init__(self,representation,objectivesNames,populationSize):
        self.representation = representation
        self.objectivesNames = objectivesNames
        self.nbObjectives = len(objectivesNames)
        self.populationSize = populationSize
        self.scores = np.array([np.array([0.0 for i in range(self.nbObjectives)]) for j in range(self.populationSize)])
        self.paretoFront = np.array([])

    def Supp(self,individual,data):
        supp = 0
        for i in range(data.shape[0]):
            if np.sum(data[i][individual]) == len(individual):
                supp+=1
        return supp/data.shape[0]


    def Support(self,indexRule,data):
        return SuppGPU(indexRule, data)

    def Confidence(self,indexRule,indexAntecedent,data):
        suppAntecedent = SuppGPU(indexAntecedent, data)
        suppRule = SuppGPU(indexRule,data)
        if suppAntecedent == 0 :
            return 0
        else:
            return suppRule/suppAntecedent

    def Comprehensibility(self,indexRule,indexConsequent):
        return np.log(1+len(indexConsequent))/np.log(1+len(indexRule))

    def Lift(self,indexRule,indexAntecedent,indexConsequent,data):
        suppAntecedent = SuppGPU(indexAntecedent, data)
        suppRule = SuppGPU(indexRule, data)
        suppConsequent = SuppGPU(indexConsequent, data)
        if suppConsequent == 0 or suppAntecedent == 0:
            return 0
        else:
            return suppRule/(suppAntecedent*suppConsequent)

    def Accuracy(self,indexRule,data):
        suppRule = SuppGPU(indexRule, data)
        suppNegativeRule = negativeSuppGPU(indexRule,data)
        return suppRule + suppNegativeRule

    def Klosgen(self,indexRule,indexAntecedent,indexConsequent,data):
        suppAntecedent = SuppGPU(indexAntecedent, data)
        suppRule = SuppGPU(indexRule, data)
        suppConsequent = SuppGPU(indexConsequent, data)
        if suppAntecedent == 0:
            return 0
        else:
            return np.sqrt(suppRule)*((suppRule/suppAntecedent)-suppConsequent)

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

    def ComputeScoreIndividual(self,individual,data):
        score = [0 for _ in range(self.nbObjectives)]
        for j in range(self.nbObjectives):
            objective = self.objectivesNames[j]
            indexRule, indexAntecedent, indexConsequent = self.GetIndividualRepresentation(individual)
            if objective == 'support':
                score[j] = self.Support(indexRule, data)
            if objective == 'confidence':
                score[j] = self.Confidence(indexRule, indexAntecedent, data)
            if objective == 'comprehensibility':
                score[j] = self.Comprehensibility(indexRule, indexConsequent)
            if objective == 'lift':
                score[j] = self.Lift(indexRule,indexAntecedent,indexConsequent,data)
            if objective == 'accuracy':
                score[j] = self.Accuracy(indexRule,data)
            if objective == 'klosgen':
                score[j] = self.Klosgen(indexRule,indexAntecedent,indexConsequent,data)
        return np.array(score)

    def ComputeScorePopulation(self,population,data):
        for i in range(len(population)):
            self.scores[i] = np.array(self.ComputeScoreIndividual(population[i],data))
        self.scores = np.array(self.scores)

    def Domination(self,a,b):
        if all(a>= b ) and any(a > b):
            return -1
        elif all(b>= a ) and any(b > a):
            return 1
        else:
            return 0

    def GetParetoFront(self):
        self.paretoFront = []
        n=[0 for i in range(self.populationSize)]
        for p in range(self.populationSize):
            for q in range(self.populationSize):
                if  self.Domination(self.scores[p], self.scores[q]) == 1:
                    n[p] += 1
            if n[p] == 0:
                self.paretoFront.append(self.scores[p])
        self.paretoFront = np.array(self.paretoFront)
        self.paretoFront = np.unique(self.paretoFront,axis=0)





