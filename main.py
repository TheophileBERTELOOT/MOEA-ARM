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
from src.Algorithm.MOFPAARM import *
from src.Algorithm.MOALOARM import *
from src.Algorithm.MODAARM import *
from src.Algorithm.MOHSBOTSARM import *

from src.Utils.Performances import *
from src.Utils.Data import *
from src.Utils.HyperParameters import *

'''
populationSize = 200
nbIteration = 10,
objectiveNames = ['support','confidence','klosgen']
parameterNames = ['s','a','c','f','e','w']
d = Data('Data/Transform/congress.csv',header=0,indexCol=0)
d.ToNumpy()
modaarm = MODAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
hyper = HyperParameters(parameterNames)
hyper.RandomSearch(30,modaarm,d.data)
hyper.SaveBestParameters('HyperParameters/MODAARM/bestParameters.json')
'''

nbIteration = 20
populationSize = 200
objectiveNames = ['support','confidence','klosgen']
criterionList = ['scores','execution time']
algorithmNameList = ['MODAARM','MOHSBOTSARM']
#algorithmNameList = ['CSOARM','mopso','nsgaii','hmofaarm','mowsaarm','mocatsoarm','motlboarm','mofpaarm','moaloarm','modaarm']

perf = Performances(algorithmNameList,criterionList,objectiveNames)
d = Data(artificial=True)
d = Data('Data/Transform/abalone.data',header=0,indexCol=0)
#d.TransformToHorizontalBinary()
#d.Save('Data/Transform/abalone.data')
d.ToNumpy()

mopso = MOPSO(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
#mobarm = MOBARM(d.data.shape[1],populationSize,nbIteration,10,len(objectiveNames),objectiveNames)
nsgaii = NSGAII(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
csoarm = CSOARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
hmofaarm = HMOFAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
mosaarm = MOSAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
mowsaarm = MOWSAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
mocatsoarm = MOCatSOARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
motlboarm = MOTLBOARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames)
mofpaarm = MOFPAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
moaloarm = MOALOARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
modaarm = MODAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
mohsbotsarm = MOHSBOTSARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)


#algorithmList = [csoarm,mopso,nsgaii,hmofaarm,mowsaarm,mocatsoarm,motlboarm,mofpaarm,moaloarm,modaarm]
algorithmList = [modaarm,mohsbotsarm]
for i in range(nbIteration):
    k = 0
    for alg in algorithmList:
        alg.Run(d.data,i)
        alg.fitness.GetParetoFront()
        perf.UpdatePerformances(score=alg.fitness.paretoFront,executionTime=alg.executionTime,i=i,algorithmName=algorithmNameList[k])
        print(algorithmNameList[k])
        print(alg.fitness.paretoFront)
        k+=1
    graph = Graphs(objectiveNames,perf.scores,path='./Figures/Comparison/paretoFront'+str(i),display=True)
    graph.GraphScores()
    perf.UpdateLeaderBoard()
    perf.FreeScores()
graph = Graphs(['execution Time'],perf.executionTime,path='./Figures/Comparison/execution_time',display=True)
graph.GraphExecutionTime()






