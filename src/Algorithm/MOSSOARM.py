from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *
import heapq

class MOSSOARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['PF']),r=3):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.indexFemale = int((0.9-rd.random()*0.25)*populationSize)
        self.r = r
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.bestInd = 0
        self.bestIndScore = np.zeros(nbObjectifs, dtype=float)
        self.worstInd = 0
        self.worstIndScore = np.zeros(nbObjectifs, dtype=float)
        self.median = 0
        self.PF = hyperParameters.hyperParameters['PF']
        self.distance = np.zeros((populationSize,populationSize),dtype=float)
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)


    def CalculDistance(self):
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                dst = distance.euclidean(self.population.population[i], self.population.population[j])
                self.distance[i, j] = dst

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
        self.bestInd = index
        self.bestIndScore = copy.deepcopy(self.fitness.scores[index])
        self.worstInd = worstIndex
        self.worstIndScore = copy.deepcopy(self.fitness.scores[worstIndex])

    def CooperativeFemaleSpider(self):
        for i in range(self.indexFemale):
            r = rd.random()
            distance = self.distance[i]
            idTest = 1
            smallest = heapq.nsmallest(idTest, distance)[-1]
            indexVibci = np.where(distance == smallest)[0][0]
            domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[indexVibci])
            while domination != 1 and idTest<self.population.populationSize and idTest ==i:
                idTest += 1
                smallest = heapq.nsmallest(idTest, distance)[-1]
                indexVibci = np.where(distance == smallest)[0][0]
                domination = self.fitness.Domination(self.fitness.scores[i], self.fitness.scores[indexVibci])
            if domination != 1:
                indexVibci = rd.randint(0,self.population.populationSize-1)
            wb = (sum(self.fitness.scores[self.bestInd])-sum(self.worstIndScore))/(sum(self.bestIndScore)-sum(self.worstIndScore))
            wb = self.population.CheckDivide0(wb)
            wc = (sum(self.fitness.scores[indexVibci]) - sum(self.worstIndScore)) / (
                        sum(self.bestIndScore) - sum(self.worstIndScore))
            wc = self.population.CheckDivide0(wc)
            Vibci = wc * np.exp(-pow(self.distance[i, indexVibci], 2))
            Vibbi = wb * np.exp(-pow(self.distance[i, self.bestInd], 2))
            alpha = rd.random()
            beta = rd.random()
            delta = rd.random()
            if r>self.PF:
                self.population.population[i] = self.population.population[i] + alpha*Vibci*(self.population.population[indexVibci] - self.population.population[i]) + beta*Vibbi*(self.population.population[self.bestInd]-self.population.population[i])+delta*(rd.random()*2)-1
            else:
                self.population.population[i] = self.population.population[i] - alpha * Vibci * (
                            self.population.population[indexVibci] - self.population.population[i]) - beta * Vibbi * (
                                                            self.population.population[self.bestInd] -
                                                            self.population.population[i]) + delta * (
                                                            rd.random() * 2) - 1
    def MatingOperator(self,maleIndex,data):
        candidate = self.distance[:self.indexFemale]
        candidate = candidate[candidate<self.r]
        if len(candidate) >0:
            indexs = np.arange(len(candidate))
            paretoFront = np.ones(len(candidate))
            for i in range(len(candidate)):
                for j in range(len(candidate)):
                    domination = self.fitness.Domination(self.fitness.scores[i], self.fitness.scores[j])
                    if domination == 1:
                        paretoFront[i] += 1
                        break
            candidate = indexs[paretoFront == 1]
            index = rd.choice(candidate)
            snew = self.population.population[maleIndex] + self.population.population[index]
            score = self.fitness.ComputeScoreIndividual(snew,data)
            domination = self.fitness.Domination(self.worstIndScore,score)
            if domination == 1:
                self.population.population[self.worstInd] = copy.deepcopy(snew)
                self.UpdateBestWorst()


    def CooperativeMaleSpider(self,data):
        weight = copy.deepcopy(self.fitness.scores)
        weight = np.sum(weight,axis=1)
        weight = weight-sum(self.worstIndScore)
        weight = weight/(sum(self.bestIndScore-self.worstIndScore))
        weight = self.population.CheckDivide0(weight)
        maleWeight = weight[self.indexFemale:]
        self.median = np.median(maleWeight)
        for i in range(self.population.populationSize-self.indexFemale):
            distance = self.distance[self.indexFemale+i][:self.indexFemale]
            alpha = rd.random()
            delta = rd.random()
            if weight[i]>self.median:
                indexF = np.argmin(distance)
                wf = (sum(self.fitness.scores[indexF])-sum(self.worstIndScore))/(sum(self.bestIndScore)-sum(self.worstIndScore))
                Vf = wf * np.exp(-pow(self.distance[i, indexF], 2))
                self.population.population[self.indexFemale+i] =self.population.population[self.indexFemale+i] + alpha *Vf*(self.population.population[indexF]-self.population.population[self.indexFemale+i]) + delta * (
                                                            rd.random() * 2) - 1
                self.MatingOperator(self.indexFemale+i,data)

            else:
                average = np.average(self.population.population[self.indexFemale:],axis=0)
                self.population.population[self.indexFemale+i] = self.population.population[self.indexFemale+i]+alpha*average

    def ResetPopulation(self, data, hyperParameters):
        self.population.InitPopulation()
        self.PF = hyperParameters.hyperParameters['PF']
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]
        self.fitness.ComputeScorePopulation(self.population.population, data)

    def Run(self,data,i):

        t1 = time()

        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.UpdateBestWorst()
        self.CalculDistance()
        self.CooperativeFemaleSpider()
        self.CooperativeMaleSpider(data)
        self.executionTime = time() - t1



