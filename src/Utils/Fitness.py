import random as rd
import numpy as np

class Fitness:
    def __init__(self,representation,objectivesNames,populationSize):
        self.representation = representation
        self.objectivesNames = objectivesNames
        self.nbObjectives = len(objectivesNames)
        self.populationSize = populationSize
        self.scores = np.array([np.array([0.0 for i in range(self.nbObjectives)]) for j in range(self.populationSize)])


    def Supp(self,individual,data):
        return sum([np.sum(np.isin(individual, row.nonzero())) == len(individual) for row in data]) / data.shape[0]


    def Support(self,indexRule,data):
        return self.Supp(indexRule, data)

    def Confidence(self,indexRule,indexAntecedent,data):
        suppAntecedent = self.Supp(indexAntecedent, data)
        suppRule = self.Supp(indexRule,data)
        if suppAntecedent == 0 :
            return 0
        else:
            return suppRule/suppAntecedent

    def Comprehensibility(self,indexRule,indexConsequent):
        return np.log(1+len(indexConsequent))/np.log(1+len(indexRule))

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




