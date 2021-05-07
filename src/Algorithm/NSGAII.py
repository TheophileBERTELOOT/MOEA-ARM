from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd

class NSGAII:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,mutationRate = 0.1, crossOverRate = 0.5):
        self.P = Population('horizontal_binary',populationSize,nbItem)
        self.Q = Population('horizontal_binary',populationSize,nbItem)
        self.R = Population('horizontal_binary',populationSize*2,nbItem)
        self.nbItem = nbItem
        self.distances = []
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary',objectiveNames,populationSize*2)
        self.fitnessFirstGeneration = Fitness('horizontal_binary',objectiveNames,populationSize)
        self.mutationRate = mutationRate
        self.crossOverRate = crossOverRate
        self.rank = [0 for _ in range(self.R.populationSize)]


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
                    offsprings.append(self.P.population[p])
                else :
                    offsprings.append(self.P.population[q])
        self.Q.SetPopulation(np.array(offsprings))

    def CrossOver(self,population):
        offsprings = []
        for i in range(int(population.populationSize*self.crossOverRate)):
            p1 = rd.randint(0,len(population.population)-1)
            p2 = rd.randint(0, len(population.population) - 1)
            index = rd.randint(1,len(population.population[p1])-1)
            offspring = np.concatenate([population.population[p1][:index],population.population[p2][index:]],axis=0)
            offsprings.append(offspring)
        offsprings = np.array(offsprings)
        offsprings = np.concatenate([population.population,np.array(offsprings)],axis=0)
        population.SetPopulation(offsprings)

    def Mutation(self,population):
        for i in range(population.populationSize):
            r = rd.random()
            if r<self.mutationRate:
                for j in range(self.nbItem*2):
                    r = rd.random()
                    if r <self.mutationRate:
                        population.population[i][j] = (rd.random()*2)-1

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
        front = self.R.population[self.rank==lastAddedFront]
        scores = np.array(self.fitness.scores[self.rank==lastAddedFront])
        front,scores,distances = self.CrowdingDistanceAssignment(front,scores)
        nbToAdd = self.P.populationSize-lastIndexInd
        currentPopulation = np.concatenate([self.R.population[:lastIndexInd],front[:nbToAdd]],axis=0)
        currentPopulationScores = np.concatenate([self.fitness.scores[:lastIndexInd],scores[:nbToAdd]],axis=0)
        currentPopulation,currentPopulationScores,self.distances = self.CrowdingDistanceAssignment(currentPopulation,currentPopulationScores)
        self.P.SetPopulation(np.array(currentPopulation))

    def PrintGraph(self,i):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.xlim([0,1])
        plt.ylim([0, 1])
        data = pd.DataFrame(self.fitness.scores,columns=['supp','confiance','comprehensibility'])
        x = self.fitness.scores[:,0]
        y = self.fitness.scores[:,1]
        z = self.fitness.scores[:,2]
        ax.set_xlabel('support')
        ax.set_ylabel('confiance')
        ax.set_zlabel('comprehensibility')
        ax.scatter(x,y,z)
        plt.show()
        fig.savefig("Figures/NSGAII/"+str(i)+".png")


    def Run(self,data):
        self.fitnessFirstGeneration.ComputeScorePopulation(self.P.population, data)
        self.BinaryTournament(True)
        self.CrossOver(self.Q)
        self.Mutation(self.Q)
        for i in range(self.nbIteration):
            self.R.SetPopulation(np.concatenate([self.P.population,self.Q.population],axis=0))
            self.fitness.ComputeScorePopulation(self.R.population, data)
            self.FastNonDominatedSort(self.R)
            self.SelectCurrentPopulation()
            self.BinaryTournament(False)
            self.CrossOver(self.Q)
            self.Mutation(self.Q)
            self.PrintGraph(i)








