from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class NSHSDEARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 hyperParameters = HyperParameters(['F','PAR'])):
        self.mincst = 0.0001
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.harmoni = Population('horizontal_binary', int(populationSize/2), nbItem)
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize,nbItem )
        self.F = hyperParameters.hyperParameters['F']
        self.Fw =  1/(self.nbItem*2)
        self.PAR =  hyperParameters.hyperParameters['PAR']
        self.mutatedVectors = np.zeros((populationSize,nbItem*2),dtype=float)+self.mincst
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.executionTime = 0
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.rank = np.zeros(populationSize,dtype=int)+self.mincst

    def CrowdingDistanceAssignment(self,front,scores):
        l = len(front)
        distances = [0 for i in range(l)]
        for i in range(self.nbObjectifs):
            front = list(zip(scores[:,i],front))
            front = np.array(sorted(front,key=lambda x:x[0]),dtype="object")
            score = np.stack(front[:,0],axis=0)
            front = np.stack(front[:,1],axis=0)
            distances[0] = np.infty
            distances[l-1] =np.infty
            j = 0
            for j in range(1,l-2):
                distances[j]+= (score[j+1]-score[j-1])/(score[0]-score[l-1]+0.00001)
            distances[j] = self.population.CheckDivide0(distances[j])
        front = list(zip(distances,front,scores))
        front = sorted(front, key=lambda x: x[0],reverse=True)
        front = np.array(front,dtype="object")
        distances = np.stack(front[:,0],axis=0)
        scores = np.stack(front[:, 2],axis=0)
        front = np.stack(front[:, 1],axis=0)
        return front,scores,distances

    def SelectCurrentPopulation(self):
        lastAddedFront = 0
        lastIndexInd = 0
        while lastIndexInd<self.harmoni.populationSize:
            lastIndexInd+=sum(self.rank==lastAddedFront)
            lastAddedFront+=1
        lastAddedFront-=1
        lastIndexInd-=sum(self.rank==lastAddedFront)
        front = copy.deepcopy(self.population.population[self.rank==lastAddedFront])
        scores = copy.deepcopy(np.array(self.fitness.scores[self.rank==lastAddedFront]))
        front,scores,distances = self.CrowdingDistanceAssignment(front,scores)
        nbToAdd = self.harmoni.populationSize-lastIndexInd
        currentPopulation = np.concatenate([self.population.population[:lastIndexInd],front[:nbToAdd]],axis=0)
        currentPopulationScores = np.concatenate([self.fitness.scores[:lastIndexInd],scores[:nbToAdd]],axis=0)
        currentPopulation,currentPopulationScores,self.distances = self.CrowdingDistanceAssignment(currentPopulation,currentPopulationScores)
        self.harmoni.SetPopulation(np.array(currentPopulation))

    def FastNonDominatedSort(self, population):
        F = []
        F1 = []
        n = [0 for _ in range(population.populationSize)]
        S = [[] for _ in range(population.populationSize)]
        for p in range(population.populationSize):
            S[p] = []
            for q in range(population.populationSize):
                if self.fitness.Domination(self.fitness.scores[p], self.fitness.scores[q]) == -1:
                    S[p].append(q)
                elif self.fitness.Domination(self.fitness.scores[p], self.fitness.scores[q]) == 1:
                    n[p] += 1
            if n[p] == 0:
                F1.append(p)
        F.append(F1)
        i = 0
        while len(F[i]) != 0:
            Q = []
            for p in range(len(F[i])):
                for q in range(len(F[i])):
                    n[q] -= 1
                    if n[q] == 0:
                        Q.append(q)
            i += 1
            F.append(Q)
        self.rank = np.ones(population.populationSize, dtype=int) * (len(F) - 1)
        for i in range(len(F)):
            for j in range(len(F[i])):
                self.rank[F[i][j]] = i
        sortedPopulation = list(zip(self.rank, population.population, self.fitness.scores))
        sortedPopulation = np.array(sorted(sortedPopulation, key=lambda x: x[0]), dtype="object")
        self.fitness.scores = np.stack(sortedPopulation[:, 2], axis=0)
        self.rank = sortedPopulation[:, 0]
        sortedPopulation = np.stack(sortedPopulation[:, 1], axis=0)
        population.SetPopulation(sortedPopulation)


    def PitchAdjustment(self):
        for i in range(self.harmoni.populationSize):
            r = rd.random()
            if r < self.PAR:
                for j in range(self.nbItem*2):
                    r = rd.random()
                    delta = self.Fw*np.random.normal()
                    s= rd.randint(-1,1)
                    if r<self.PAR:
                        self.mutatedVectors[i][j] = self.mutatedVectors[i][j]+s*delta

    def Mutation(self):
        for i in range(self.harmoni.populationSize):
            a = rd.randint(0,self.harmoni.populationSize-1)
            b = rd.randint(0,self.harmoni.populationSize-1)
            c = rd.randint(0,self.harmoni.populationSize-1)
            while a==i or b==i or c==i or a==b or a==c or b==c:
                a = rd.randint(0, self.harmoni.populationSize - 1)
                b = rd.randint(0, self.harmoni.populationSize - 1)
                c = rd.randint(0, self.harmoni.populationSize - 1)
            self.mutatedVectors[i] = self.harmoni.population[a] + self.F*(self.harmoni.population[b]-self.harmoni.population[c])

    def UpdateFw(self,i):
        c = np.log(((1/(1000*self.nbItem*2)))/(1/(self.nbItem*2)))/self.nbIteration
        self.Fw = 1*np.exp(c*i)

    def ResetPopulation(self, data, hyperParameters):
        self.population.InitPopulation()
        self.harmoni.InitPopulation()
        self.F = hyperParameters.hyperParameters['F']
        self.PAR =  hyperParameters.hyperParameters['PAR']
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]
        self.fitness.ComputeScorePopulation(self.population.population, data)


    def Run(self,data,i):
        t1 = time()
        self.mutatedVectors = -1*np.ones((int(self.population.populationSize/2),self.nbItem*2),dtype=float)
        self.Mutation()
        self.PitchAdjustment()
        self.population.SetPopulation(np.concatenate([copy.deepcopy(self.mutatedVectors),copy.deepcopy(self.harmoni.population)],axis=0))
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.FastNonDominatedSort(self.population)
        self.SelectCurrentPopulation()
        self.executionTime = time() - t1



