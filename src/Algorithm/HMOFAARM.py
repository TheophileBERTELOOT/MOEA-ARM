from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import  time
from src.Utils.HyperParameters import *

class HMOFAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 nbSolution=10,hyperParameters = HyperParameters(['alpha','beta0','crossOverRate']),
                ):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize)
        self.alpha = hyperParameters.hyperParameters['alpha']
        self.beta0 = hyperParameters.hyperParameters['beta0']
        self.gamma = 1/((self.nbItem*2)**2)
        self.crossOverRate = hyperParameters.hyperParameters['crossOverRate']
        self.nbSolution = nbSolution
        self.paretoFront = []
        self.paretoFrontScore = []
        self.executionTime = 0


    def UpdateAlpha(self,i):
        self.alpha*= (1-(i/self.nbIteration))

    def UpdateBeta0(self):
        r = rd.random()
        if r < 0.5:
            self.beta0 = rd.random()

    def UpdateIndividual(self,xid,xjd):
        epsilon = rd.random()-0.5
        distance = self.CalculDistance(xid,xjd)**2
        return xid + self.beta0*np.exp(-self.gamma*distance)*(xjd-xid)+self.alpha*epsilon

    def CrossOver(self,xit,xit1):
        ind = []
        for i in range(self.nbItem*2):
            r = rd.random()
            if r< self.crossOverRate:
                ind.append(xit1[i])
            else :
                ind.append(xit[i])
        return np.array(ind)

    def UpdatePopulation(self,data):
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                if self.fitness.Domination(self.fitness.scores[j],self.fitness.scores[i]) == -1:
                    self.UpdateBeta0()
                    xit = self.population.population[i]
                    xjd = self.population.population[j]
                    xit1 = self.UpdateIndividual(xit,xjd)
                    xit1Star = self.CrossOver(xit,xit1)
                    xit1Score = self.fitness.ComputeScoreIndividual(xit1,data)
                    xit1starScore = self.fitness.ComputeScoreIndividual(xit1Star,data)
                    domination = self.fitness.Domination(xit1Score,xit1starScore)
                    if domination == 1:
                        self.population.population[i] = copy.deepcopy(xit1Star)
                    else:
                        self.population.population[i] = copy.deepcopy(xit1)


    def CalculDistance(self,xid,xjd):
        return distance.euclidean(xid, xjd)

    def FastNonDominatedSort(self,population):
        F = []
        F1 = []
        n = [0 for _ in range(population.populationSize)]
        S = [[] for _ in range(population.populationSize)]
        self.rank = [0 for _ in range(population.populationSize)]
        for p in range(population.populationSize):
            S[p] = []
            for q in range(population.populationSize):
                if self.fitness.Domination(self.fitness.scores[p],self.fitness.scores[q]) == -1:
                    S[p].append(q)
                elif self.fitness.Domination(self.fitness.scores[p],self.fitness.scores[q]) == 1:
                    n[p]+=1
            if n[p] == 0:
                self.rank[p] = 0
                F1.append(p)
        F.append(F1)
        i = 0
        while len(F[i]) != 0 :
            Q = []
            for p in range(len(F[i])):
                for q in range(len(F[i])):
                    n[q]-=1
                    if n[q] == 0:
                        self.rank[q] = i+1
                        Q.append(q)
            i+=1
            F.append(Q)
        self.rank = np.array(self.rank)
        sortedPopulation = list(zip(self.rank,population.population,self.fitness.scores))
        sortedPopulation = np.array(sorted(sortedPopulation,key=lambda x:x[0]),dtype="object")
        self.fitness.scores = np.stack(sortedPopulation[:,2],axis=0)
        self.rank = sortedPopulation[:,0]
        sortedPopulation = np.stack(sortedPopulation[:,1],axis=0)
        population.SetPopulation(sortedPopulation)


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
            for j in range(2,l-2):
                distances[j]+= (score[j+1]-score[j-1])/(score[0]-score[l-1])
                distances[j] = self.population.CheckDivide0(distances[j])
        front = list(zip(distances,front,scores))
        front = sorted(front, key=lambda x: x[0],reverse=True)
        front = np.array(front,dtype="object")
        distances = np.stack(front[:,0],axis=0)
        scores = np.stack(front[:, 2],axis=0)
        front = np.stack(front[:, 1],axis=0)
        return front,scores,distances

    def FindParetoFront(self):
        nonDominated = self.rank==0
        candidates = self.population.population[nonDominated]
        candidatesScore = np.array(self.fitness.scores[nonDominated])
        front, scores, distances = self.CrowdingDistanceAssignment(candidates, candidatesScore)
        self.paretoFront = copy.deepcopy(front[:self.nbSolution])
        self.paretoFrontScore=copy.deepcopy(scores[:self.nbSolution])

    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.alpha = hyperParameters.hyperParameters['alpha']
        self.beta0 = hyperParameters.hyperParameters['beta0']
        self.gamma = 1/((self.nbItem*2)**2)
        self.crossOverRate = hyperParameters.hyperParameters['crossOverRate']
        self.paretoFront = []
        self.paretoFrontScore = []

    def Run(self,data,i):

        t1 = time()
        self.UpdateAlpha(i)
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdatePopulation(data)
        self.FastNonDominatedSort(self.population)
        self.FindParetoFront()
        self.executionTime = time() - t1


