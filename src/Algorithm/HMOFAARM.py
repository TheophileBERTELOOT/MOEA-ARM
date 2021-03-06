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
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize,nbItem)
        self.alpha = hyperParameters.hyperParameters['alpha']
        self.beta0 = hyperParameters.hyperParameters['beta0']
        self.betaInit = hyperParameters.hyperParameters['beta0']
        self.gamma = 1/((self.nbItem*2)**2)
        self.crossOverRate = hyperParameters.hyperParameters['crossOverRate']
        self.nbSolution = nbSolution
        self.executionTime = 0



    def UpdateAlpha(self,i):
        self.alpha*= (1-(i/self.nbIteration))

    def UpdateBeta0(self):
        r = rd.random()
        if r < 0.5:
            self.beta0 = rd.random()
        else:
            self.beta0 = self.betaInit


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
            for j in np.random.randint(self.population.populationSize,size=10):
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


    def ResetPopulation(self,data,hyperParameters):
        self.population.InitPopulation()
        self.alpha = hyperParameters.hyperParameters['alpha']
        self.beta0 = hyperParameters.hyperParameters['beta0']
        self.gamma = 1/((self.nbItem*2)**2)
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]
        self.crossOverRate = hyperParameters.hyperParameters['crossOverRate']


    def SelectCurrentPopulation(self):
        lastAddedFront = 0
        lastIndexInd = 0
        while lastIndexInd<self.population.populationSize:
            lastIndexInd+=sum(self.rank==lastAddedFront)
            lastAddedFront+=1
        lastAddedFront-=1
        lastIndexInd-=sum(self.rank==lastAddedFront)
        front = copy.deepcopy(self.population.population[self.rank==lastAddedFront])
        scores = copy.deepcopy(np.array(self.fitness.scores[self.rank==lastAddedFront]))
        front,scores,distances = self.CrowdingDistanceAssignment(front,scores)
        nbToAdd = self.population.populationSize-lastIndexInd
        currentPopulation = np.concatenate([self.population.population[:lastIndexInd],front[:nbToAdd]],axis=0)
        currentPopulationScores = np.concatenate([self.fitness.scores[:lastIndexInd],scores[:nbToAdd]],axis=0)
        currentPopulation,currentPopulationScores,self.distances = self.CrowdingDistanceAssignment(currentPopulation,currentPopulationScores)
        self.population.SetPopulation(np.array(currentPopulation))

    def Run(self,data,i):
        t1 = time()
        self.UpdateAlpha(i)
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdatePopulation(data)
        self.FastNonDominatedSort(self.population)
        self.SelectCurrentPopulation()
        self.executionTime = time() - t1



