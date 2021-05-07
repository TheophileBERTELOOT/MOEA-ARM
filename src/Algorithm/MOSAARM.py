from src.Utils.Fitness import *
from src.Utils.Population import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance

class MOSAARM:
    def __init__(self,nbItem,populationSize,nbIteration,nbObjectifs,objectiveNames,tempInitial = 1,nbIterationPerTemp=100,nbChanges=2,alpha = 0.9):
        self.population = Population('horizontal_binary', populationSize, nbItem)
        self.nbItem = nbItem
        self.nbIteration = nbIteration
        self.nbObjectifs = nbObjectifs
        self.fitness = Fitness('horizontal_binary', objectiveNames, populationSize)
        self.tempInitial = tempInitial
        self.nbIterationPerTemp = nbIterationPerTemp
        self.nbChanges = nbChanges
        self.alpha = alpha

    def GenerateRule(self,i):
        ind = copy.deepcopy(self.population.population[i])
        for j in range(self.nbChanges):
            index = rd.randint(0,(self.nbItem*2)-1)
            ind[index]*=-1
        return ind

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
        fig.savefig("Figures/MOSAARM/"+str(i)+".png")

    def Run(self,data):
        T = self.tempInitial
        for k in range(self.nbIteration):
            self.fitness.ComputeScorePopulation(self.population.population,data)
            for j in range(self.population.populationSize):
                for m in range(self.nbIterationPerTemp):
                    newRule = self.GenerateRule(j)
                    newRuleScore = self.fitness.ComputeScoreIndividual(newRule,data)
                    domination = self.fitness.Domination(self.fitness.scores[j],newRuleScore)
                    delta =   abs(sum(newRuleScore)-sum(self.fitness.scores[j]))
                    if domination == 1:
                        self.population.population[j] = copy.deepcopy(newRule)
                    else:
                        r = rd.random()
                        if np.exp(-delta/T)>r and delta != 0 and sum(newRuleScore) !=0:
                            self.population.population[j] = copy.deepcopy(newRule)
            T*=self.alpha
            print(self.fitness.scores)
            self.PrintGraph(k)
