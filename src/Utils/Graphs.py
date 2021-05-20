import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

class Graphs:
    def __init__(self,objectiveNames,data,save=True,display=True,path='./Figures/'):
        self.objectiveNames = objectiveNames
        self.data = data
        self.save = save
        self.path = path
        self.display = display
        self.CheckIfPathExist()

    def CheckIfPathExist(self):
        p = self.path.split('/')
        p = p[:-1]
        p = '/'.join(p)
        pathExist = os.path.exists(p)
        if not pathExist :
            os.mkdir(p)

    def GraphNbRules(self):
        fig = plt.figure()
        sns.barplot(x='algorithm', y='nbRules', data=self.data)
        plt.xticks(rotation=70)
        plt.tight_layout()
        if self.display:
            plt.show()
        if self.save:
            fig.savefig(self.path + ".png")

    def GraphExecutionTime(self):
        fig = plt.figure()
        sns.lineplot(x='i',y='execution Time',hue='algorithm',data=self.data)
        if self.display:
            plt.show()
        if self.save:
            fig.savefig(self.path+".png")

    def GraphScores(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel(self.objectiveNames[0])
        ax.set_ylabel(self.objectiveNames[1])
        ax.set_zlabel(self.objectiveNames[2])

        for alg in self.data.algorithm.unique():
            ax.scatter(self.data[self.data.algorithm==alg][self.objectiveNames[0]],
                       self.data[self.data.algorithm==alg][self.objectiveNames[1]],
                       self.data[self.data.algorithm==alg][self.objectiveNames[2]],
                       label=alg)
        ax.legend()

        if self.display:
            plt.show()
        if self.save:
            fig.savefig(self.path+".png")




    def Graph3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        ax.set_xlabel(self.objectiveNames[0])
        ax.set_ylabel(self.objectiveNames[1])
        ax.set_zlabel(self.objectiveNames[2])
        ax.scatter(x, y, z)
        if self.display:
            plt.show()
        if self.save:
            fig.savefig(self.path+".png")


