from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from src.Utils.Graphs import *
from time import time
import numpy as np
import scipy.special
from sklearn.preprocessing import MinMaxScaler

class MODAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,data,
                 minDist = 8,nbChanges = 5,s=0.5,a=0.5,c=0.5,f=0.5,e=0.05,w=0.3,
                 save=True,display=True,path='Figures/'):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize )
        self.food = np.zeros
        self.predator = []
        self.executionTime = 0
        self.minDist = minDist
        self.nbChanges= nbChanges
        self.s=s
        self.a=a
        self.c=c
        self.f=f
        self.e=e
        self.w=w
        self.distance = np.zeros((populationSize,populationSize),dtype=float)
        self.velocity = np.zeros((populationSize,nbItem*2),dtype=float)
        self.orientaiton = np.zeros((populationSize,nbItem*2),dtype=float)
        self.fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdatePredator()
        self.UpdateFood()
        self.CalculDistance()

    def UpdateVelocity(self):
        self.velocity = np.zeros((self.population.populationSize, self.nbItem * 2))
        for i in range(self.population.populationSize):
            nbChange= rd.randint(1,self.nbChanges)
            for j in range(nbChange):
                index = rd.randint(0,self.nbItem*2-1)
                self.velocity[i,index] = float(rd.randint(-1,1))

    def GetNeighbors(self,df):
        N = []
        for i in range(self.population.populationSize):
            if self.distance[i,df] <= self.minDist:
                N.append(i)
        return N

    def UpdateOrientation(self):
        for i in range(self.population.populationSize):
            N = self.GetNeighbors(i)
            A = self.Alignment(N)
            S = self.Separation(i,N)
            C = self.Cohesion(i,N)
            F = self.FoodAttraction(i)
            E = self.PredatorDistraction(i)
            self.orientaiton[i] = (self.s*S + self.a*A+self.c *C+self.f*F+self.e*E)+ self.w*self.orientaiton[i]




    def Separation(self,df,N):
        S =np.zeros(self.nbItem*2,dtype=float)
        for j in range(len(N)):
            S= S +self.population.population[df]-self.population.population[N[j]]
        return -S

    def Alignment(self,N):
        self.UpdateVelocity()
        A = np.zeros(self.nbItem*2,dtype=float)
        for i in range(len(N)):
            A = A +self.velocity[N[i]]
        return A/len(N)

    def Cohesion(self,df,N):
        C = np.zeros(self.nbItem*2,dtype=float)
        for i in range(len(N)):
            C = C +self.population.population[N[i]]
        C = C/len(N)
        return C-self.population.population[df]

    def FoodAttraction(self,df):
        return self.food - self.population.population[df]

    def PredatorDistraction(self,df):
        return self.predator + self.population.population[df]

    def CalculDistance(self):
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize) :
                dst = distance.euclidean(self.population.population[i], self.population.population[j])
                self.distance[i,j] = dst

    def UpdatePredator(self):
        paretoFront = np.zeros(self.population.populationSize)
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] = paretoFront[i] +1
        index = np.argmax(paretoFront)
        self.predator = copy.deepcopy(self.population.population[index])

    def UpdateFood(self):
        indexs = np.arange(self.population.populationSize)
        paretoFront = np.ones(self.population.populationSize)
        for i in range(self.population.populationSize):
            for j in range(self.population.populationSize):
                domination = self.fitness.Domination(self.fitness.scores[i],self.fitness.scores[j])
                if domination == 1:
                    paretoFront[i] = 0
                    break
        candidate = indexs[paretoFront == 1]
        index = rd.choice(candidate)
        self.food = copy.deepcopy(self.population.population[index])

    def gamma(self,x):
        return scipy.special.factorial(x-1)

    def RandomWalk(self,df):
        beta = 1.5
        sigma = np.power((self.gamma(1+beta )*np.sin(np.pi*beta/2))/(self.gamma((1+beta)/2)*beta*np.power(2,(beta-1)/2)),1/beta)
        levy = 0.01* (rd.random()*sigma)/(np.power(rd.random(),1/beta))
        self.population.population[df] = self.population.population[df] + rd.randint(-1,1)*levy * self.population.population[df]

    def Run(self,data,i):
        t1 = time()
        self.minDist -=0.1
        self.UpdateOrientation()
        for j in range(self.population.populationSize):
            N = self.GetNeighbors(j)
            if len(N)>0:
                self.population.population[j] = self.population.population[j] + self.orientaiton[j]
            else:
                self.RandomWalk(j)
        self.fitness.ComputeScorePopulation(self.population.population,data)
        self.UpdateFood()
        self.UpdatePredator()
        self.executionTime = time() - t1
        self.CalculDistance()