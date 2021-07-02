import random as rd
import numpy as np
from numba import jit, cuda
from numba import njit
from src.Utils.Population import *
from scipy.spatial import distance

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
        self.population = Population(representation,0,0)
        self.scores = np.array([np.array([0.0 for i in range(self.nbObjectives)]) for j in range(self.populationSize)])
        self.paretoFront = np.zeros((1,len(objectivesNames)),dtype=float)
        self.paretoFrontSolutions = []
        self.distances = []
        self.averageDistances = 0
        self.coverage = 0


    def Supp(self,individual,data):
        supp = 0
        for i in range(data.shape[0]):
            if np.sum(data[i][individual]) == len(individual):
                supp+=1
        return supp/data.shape[0]


    def Support(self,indexRule,data):
        return SuppGPU(indexRule, data)

    def Confidence(self,suppRule,suppAntecedent):
        if suppAntecedent == 0 :
            return 0
        else:
            return suppRule/suppAntecedent

    def Comprehensibility(self,indexRule,indexConsequent):
        return np.log(1+len(indexConsequent))/np.log(1+len(indexRule))

    def Lift(self,suppRule,suppAntecedent,suppConsequent):
        if suppConsequent == 0 or suppAntecedent == 0:
            return 0
        else:
            return suppRule/(suppAntecedent*suppConsequent)

    def Accuracy(self,suppRule,suppNegativeRule):
        return suppRule + suppNegativeRule

    def Klosgen(self,suppRule,suppAntecedent,suppConsequent):
        if suppAntecedent == 0:
            return 0
        else:
            return np.sqrt(suppRule)*((suppRule/suppAntecedent)-suppConsequent)

    def Cosine(self, suppRule, suppAntecedent, suppConsequent):
        if suppAntecedent == 0 or suppConsequent == 0 or suppRule == 0:
            return 0
        else:
            return suppRule/np.sqrt(suppAntecedent*suppConsequent)

    def Jaccard(self,suppRule, suppAntecedent, suppConsequent):
        if suppRule == 0 or suppAntecedent+suppConsequent == suppRule :
            return 0
        else:
            return suppRule/(suppAntecedent+suppConsequent-suppRule)

    def PiatetskiShapiro(self,suppRule, suppAntecedent, suppConsequent):
        return suppRule - suppAntecedent*suppConsequent

    def ComputeScoreIndividual(self,individual,data):
        score = [0 for _ in range(self.nbObjectives)]
        indexRule, indexAntecedent, indexConsequent = self.population.GetIndividualRepresentation(individual)
        suppRule = SuppGPU(indexRule, data)
        suppAntecedent = SuppGPU(indexAntecedent, data)
        suppConsequent = SuppGPU(indexConsequent, data)
        for j in range(self.nbObjectives):
            objective = self.objectivesNames[j]
            if objective == 'support':
                score[j] = suppRule
            if objective == 'confidence':
                score[j] = self.Confidence(suppRule, suppAntecedent)
            if objective == 'comprehensibility':
                score[j] = self.Comprehensibility(indexRule, indexConsequent)
            if objective == 'lift':
                score[j] = self.Lift(suppRule,suppAntecedent,suppConsequent)
            if objective == 'accuracy':
                suppNegativeRule = negativeSuppGPU(indexRule,data)
                score[j] = self.Accuracy(suppRule,suppNegativeRule)
            if objective == 'klosgen':
                score[j] = self.Klosgen(suppRule,suppAntecedent,suppConsequent)
            if objective == 'cosine':
                score[j] = self.Cosine(suppRule, suppAntecedent, suppConsequent)
            if objective == 'jaccard':
                score[j] = self.Jaccard(suppRule, suppAntecedent, suppConsequent)
            if objective == 'piatetskiShapiro':
                score[j] = self.PiatetskiShapiro(suppRule, suppAntecedent, suppConsequent)

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
        temp = np.concatenate([self.scores,copy.deepcopy(self.paretoFront)])
        self.paretoFront = []
        for p in range(len(temp)):
            dominate = True
            for q in range(len(temp)):
                if  self.Domination(temp[p], temp[q]) == 1:
                    dominate = False
                    break
            if dominate:
                self.paretoFront.append(temp[p])
        self.paretoFront = np.array(self.paretoFront)
        self.paretoFront = np.unique(self.paretoFront,axis=0)

    def GetHead(self,sizeHead,population):
        if self.paretoFrontSolutions != []:
            tempSolutions = np.concatenate([self.paretoFrontSolutions, copy.deepcopy(population.GetPopulationRepresentation())])
        else:
            tempSolutions =copy.deepcopy(population.GetPopulationRepresentation())
        indexUniques = np.unique(tempSolutions,return_index=True,axis=0)[1]
        temp = np.concatenate( [copy.deepcopy(self.paretoFront),self.scores])
        sumScore = np.sum(temp[indexUniques],axis=1)
        if len(sumScore)<sizeHead:
            self.paretoFront = temp[indexUniques]
            self.paretoFrontSolutions = tempSolutions[indexUniques]
        else:
            indexParetoFront = np.argpartition(sumScore,-sizeHead)[-sizeHead:]
            self.paretoFront = temp[indexUniques][indexParetoFront]
            self.paretoFrontSolutions = tempSolutions[indexUniques][indexParetoFront]

    def GetUniquePop(self,population):
        tempSolutions =copy.deepcopy(population.GetPopulationRepresentation())
        indexUniques = np.unique(tempSolutions, return_index=True, axis=0)[1]
        self.paretoFront = self.scores[indexUniques]
        self.paretoFrontSolutions = tempSolutions[indexUniques]


    def GetDistances(self,population):
        self.averageDistances = 0
        self.distances = np.zeros(len(self.paretoFrontSolutions),dtype=float)
        for i in range(len(self.distances)):
            indAvgDistance = 0
            indi = np.array(population.population[i]>0,dtype=int)
            for j in range(len(self.distances)):
                indj = np.array(population.population[j]>0,dtype=int)
                if i != j:
                    indAvgDistance+=distance.euclidean(indi, indj)
            indAvgDistance/=(len(self.distances)-1)
            self.distances[i] = indAvgDistance
        self.averageDistances = np.average(self.distances)

    def GetCoverage(self,data,population):
        coverage = 0
        for i in range(data.shape[0]):
            for j in range(population.populationSize):
                indexRule, _, _ = population.GetIndividualRepresentation(population.population[j])
                if np.sum(data[i][indexRule]) == len(indexRule):
                    coverage+=1
                    break
        self.coverage =  coverage / data.shape[0]










