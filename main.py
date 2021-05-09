import pandas as pd

from src.Algorithm.MOPSO import *
from src.Algorithm.MOBARM import *
from src.Algorithm.NSGAII import *
from src.Algorithm.CSOARM import *
from src.Algorithm.HMOFAARM import *
from src.Algorithm.MOSAARM import *
from src.Algorithm.MOWSAARM import *
from src.Algorithm.MOCatSOARM import *

from src.Utils.Performances import *

nbIteration = 20
populationSize = 50
objectiveNames = ['support','confidence','lift']
criterionList = ['scores','execution time']
algorithmNameList = ['MOWSAARM','MOCatSOARM','NSGAII']

perf = Performances(algorithmNameList,criterionList,objectiveNames)
data = pd.read_csv('Data/bankrupt.csv')
data = data.to_numpy()
#mopso = MOPSO(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
#mobarm = MOBARM(data.shape[1],populationSize,nbIteration,10,len(objectiveNames),objectiveNames,save=True,display=True,path='Figures/MOBARM/')
#nsgaii = NSGAII(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,data,save=True,display=False,path='Figures/NSGAII/')
#csoarm = CSOARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,data)
#hmofaarm = HMOFAARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
#mosaarm = MOSAARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=True,path='Figures/MOSAARM/')
mowsaarm = MOWSAARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=False,path='Figures/MOWSAARM/')
mocatsoarm = MOCatSOARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=False,path='Figures/MOCatSOARM/')

algorithmList = [mowsaarm,mocatsoarm]

for i in range(nbIteration):
    k = 0
    for alg in algorithmList:
        alg.Run(data,i)
        perf.UpdatePerformances(score=alg.fitness.scores,executionTime=alg.executionTime,i=i,algorithmName=algorithmNameList[k])
        k+=1
    graph = Graphs(objectiveNames,perf.scores)
    graph.GraphScores()
    perf.FreeScores()
graph = Graphs(['execution Time'],perf.executionTime)
graph.GraphExecutionTime()






