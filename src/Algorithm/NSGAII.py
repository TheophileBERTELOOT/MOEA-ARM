from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from src.Utils.Graphs import *
from time import time
from src.Utils.HyperParameters import *

class NSGAII:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,
                 objectiveNames,data,hyperParameters = HyperParameters(['mutationRate', 'crossOverRate']),nbChanges=5):
        self.P = Population('horizontal_binary',int(populationSize/2),nbItem)
        self.Q = Population('horizontal_binary',int(populationSize/2),nbItem)
        self.population = Population('horizontal_binary',populationSize,nbItem)
        self.nbItem = nbItem
        self.distances = []
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.nbChanges = nbChanges
        self.fitness = Fitness('horizontal_binary',objectiveNames,populationSize,nbItem)
        self.fitnessFirstGeneration = Fitness('horizontal_binary',objectiveNames,int(populationSize/2),nbItem)
        self.mutationRate = hyperParameters.hyperParameters['mutationRate']
        self.crossOverRate = hyperParameters.hyperParameters['crossOverRate']
        self.rank = [0 for _ in range(self.population.populationSize)]
        self.executionTime = 0

        self.fitnessFirstGeneration.ComputeScorePopulation(self.P.population, data)
        self.BinaryTournament(True)
        self.CrossOver(self.Q)
        self.Mutation(self.Q)


    def FastNonDominatedSort(self,population):
        F = []
        F1 = []
        n = [0 for _ in range(population.populationSize)]
        S = [[] for _ in range(population.populationSize)]
        for p in range(population.populationSize):
            S[p] = []
            for q in range(population.populationSize):
                if self.fitness.Domination(self.fitness.scores[p],self.fitness.scores[q]) == -1:
                    S[p].append(q)
                elif self.fitness.Domination(self.fitness.scores[p],self.fitness.scores[q]) == 1:
                    n[p]+=1
            if n[p] == 0:
                F1.append(p)
        F.append(F1)
        i = 0
        while len(F[i]) != 0 :
            Q = []
            for p in range(len(F[i])):
                for q in range(len(F[i])):
                    n[q]-=1
                    if n[q] == 0:
                        Q.append(q)
            i+=1
            F.append(Q)
        self.rank = np.ones(population.populationSize,dtype=int)*(len(F)-1)
        for i in range(len(F)):
            for j in range(len(F[i])):
                self.rank[F[i][j]] = i
        sortedPopulation = list(zip(self.rank,population.population,self.fitness.scores))
        sortedPopulation = np.array(sorted(sortedPopulation,key=lambda x:x[0]),dtype="object")
        self.fitness.scores = np.stack(sortedPopulation[:,2],axis=0)
        self.rank = sortedPopulation[:,0]
        sortedPopulation = np.stack(sortedPopulation[:,1],axis=0)
        population.SetPopulation(sortedPopulation)

    def BinaryTournament(self,firstGeneration):
        offsprings = []
        for i in range(int(self.P.populationSize*(1-self.crossOverRate))):
            p = rd.randint(0,self.P.populationSize-1)
            q = rd.randint(0,self.P.populationSize-1)
            if firstGeneration:
                if self.fitness.Domination(self.fitness.scores[p],self.fitness.scores[q]) == -1:
                    offsprings.append(self.P.population[p])
                elif self.fitness.Domination(self.fitness.scores[p],self.fitness.scores[q]) == 1:
                    offsprings.append(self.P.population[q])
                else:
                    r = rd.random()
                    if r <0.5:
                        offsprings.append(self.P.population[p])
                    else:
                        offsprings.append(self.P.population[q])
            else:
                if self.distances[p]>=self.distances[q]:
                    offsprings.append(copy.deepcopy(self.P.population[p]))
                else :
                    offsprings.append(copy.deepcopy(self.P.population[q]))
        self.Q.SetPopulation(np.array(offsprings))

    def CrossOver(self,population):
        offsprings = []
        nbCross = int(population.populationSize*self.crossOverRate)
        if nbCross+len(population.population)!=self.Q.populationSize:
            nbCross+=1
        for i in range(nbCross):
            p1 = rd.randint(0,len(population.population)-1)
            p2 = rd.randint(0, len(population.population) - 1)
            offspring = np.zeros(self.nbItem*2,dtype=float)
            for j in range(self.nbItem*2):
                r = rd.random()
                if r<=0.5:
                    offspring[j] = population.population[p1][j]
                else:
                    offspring[j] = population.population[p2][j]
            offsprings.append(offspring)
        offsprings = np.concatenate([population.population,np.array(offsprings)],axis=0)
        population.SetPopulation(offsprings)


    def Mutation(self, population):
        for i in range(population.populationSize):
            r = rd.random()
            if r < self.mutationRate:
                nbChange = rd.randint(1, self.nbChanges)
                for j in range(nbChange):
                    index = rd.randint(0, self.nbItem * 2 - 1)
                    population.population[i][index] = -1*population.population[i][index]

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
        while lastIndexInd<self.P.populationSize:
            lastIndexInd+=sum(self.rank==lastAddedFront)
            lastAddedFront+=1
        lastAddedFront-=1
        lastIndexInd-=sum(self.rank==lastAddedFront)
        front = self.population.population[self.rank==lastAddedFront]
        scores = np.array(self.fitness.scores[self.rank==lastAddedFront])
        front,scores,distances = self.CrowdingDistanceAssignment(front,scores)
        nbToAdd = self.P.populationSize-lastIndexInd
        currentPopulation = np.concatenate([self.population.population[:lastIndexInd],front[:nbToAdd]],axis=0)
        currentPopulationScores = np.concatenate([self.fitness.scores[:lastIndexInd],scores[:nbToAdd]],axis=0)
        currentPopulation,currentPopulationScores,self.distances = self.CrowdingDistanceAssignment(currentPopulation,currentPopulationScores)
        self.P.SetPopulation(np.array(currentPopulation))

    def ResetPopulation(self, data, hyperParameters):
        self.P.InitPopulation()
        self.Q.InitPopulation()
        self.population.InitPopulation()
        self.mutationRate = hyperParameters.hyperParameters['mutationRate']
        self.crossOverRate = hyperParameters.hyperParameters['crossOverRate']
        self.fitnessFirstGeneration.ComputeScorePopulation(self.P.population, data)
        self.fitness.paretoFront=np.zeros((1,len(self.fitness.objectivesNames)),dtype=float)
        self.fitness.distances = []
        self.fitness.coverage = []
        self.fitness.paretoFrontSolutions=[]
        self.BinaryTournament(True)
        self.CrossOver(self.Q)
        self.Mutation(self.Q)

    def Run(self,data,i):

        t1 = time()
        self.population.SetPopulation(np.concatenate([copy.deepcopy(self.P.population),copy.deepcopy(self.Q.population)],axis=0))
        self.population.CheckIfNull()
        self.fitness.ComputeScorePopulation(self.population.population, data)
        self.FastNonDominatedSort(self.population)
        self.SelectCurrentPopulation()
        self.BinaryTournament(False)
        self.CrossOver(self.Q)
        self.Mutation(self.Q)
        self.executionTime = time() - t1








