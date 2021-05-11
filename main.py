import pandas as pd

from src.Algorithm.MOPSO import *
from src.Algorithm.MOBARM import *
from src.Algorithm.NSGAII import *
from src.Algorithm.CSOARM import *
from src.Algorithm.HMOFAARM import *
from src.Algorithm.MOSAARM import *
from src.Algorithm.MOWSAARM import *
from src.Algorithm.MOCatSOARM import *
from src.Algorithm.MOTLBOARM import *

from src.Utils.Performances import *

nbIteration = 20
populationSize = 200
objectiveNames = ['support','confidence','klosgen']
criterionList = ['scores','execution time']
#algorithmNameList = ['MOTLBOARM','MOCatSOARM']
algorithmNameList = ['CSOARM','mopso','nsgaii','hmofaarm','mowsaarm','mocatsoarm','motlboarm']

perf = Performances(algorithmNameList,criterionList,objectiveNames)
data = pd.read_csv('Data/congress.csv')
data = data.to_numpy()
mopso = MOPSO(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
#mobarm = MOBARM(data.shape[1],populationSize,nbIteration,10,len(objectiveNames),objectiveNames,save=True,display=True,path='Figures/MOBARM/')
nsgaii = NSGAII(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,data,save=True,display=False,path='Figures/NSGAII/')
csoarm = CSOARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,data,save=True,display=True,path='Figures/CSOARM/')
hmofaarm = HMOFAARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=True,path='Figures/HMOFAARM/')
mosaarm = MOSAARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=True,path='Figures/MOSAARM/')
mowsaarm = MOWSAARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=False,path='Figures/MOWSAARM/')
mocatsoarm = MOCatSOARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=False,path='Figures/MOCatSOARM/')
motlboarm = MOTLBOARM(data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,save=True,display=False,path='Figures/MOTLBOARM/')

algorithmList = [csoarm,mopso,nsgaii,hmofaarm,mowsaarm,mocatsoarm,motlboarm]
#algorithmList = [motlboarm,mocatsoarm]
for i in range(nbIteration):
    k = 0
    for alg in algorithmList:
        alg.Run(data,i)
        alg.fitness.GetParetoFront()
        perf.UpdatePerformances(score=alg.fitness.paretoFront,executionTime=alg.executionTime,i=i,algorithmName=algorithmNameList[k])
        print(algorithmNameList[k])
        print(alg.fitness.paretoFront)
        k+=1
    graph = Graphs(objectiveNames,perf.scores,path='./Figures/Comparison/paretoFront'+str(i),display=False)
    graph.GraphScores()
    perf.UpdateLeaderBoard()
    perf.FreeScores()
graph = Graphs(['execution Time'],perf.executionTime,path='./Figures/Comparison/execution_time')
graph.GraphExecutionTime()






