import pandas as pd
from src.Utils.Performances import *
from src.Utils.Data import *
from src.Utils.HyperParameters import *
from src.Utils.Graphs import *
from src.Utils.Experiments import *
from os import path,mkdir
import sys

'''
np.set_printoptions(threshold=sys.maxsize)
populationSize = 100
nbIteration = 20
objectiveNames = ['support','confidence','cosine']
parameterNames = ['alpha','beta0','crossOverRate']
p= 'HyperParameters/HMOFAARM/'
if(not path.exists(p)):
    mkdir(p)
d = Data('Data/Transform/breast-cancer.csv',header=0,indexCol=0)
d.ToNumpy()
alg = HMOFAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
hyper = HyperParameters(parameterNames,sizeHead=10)
hyper.RandomSearch(100,alg,d.data)
hyper.SaveBestParameters(p+'bestParameters.json')
'''


nbIteration = 50
sizeHead = 10
iterationInitiale = 0
nbRepetition = 50 - iterationInitiale
populationSize = 100
objectiveNames = ['support','confidence','cosine']
criterionList = ['scores','execution time','distances','coverages']
algorithmNameList = ['mosaarm','mossoarm','mocsoarm','nsgaii','hmofaarm','mowsaarm','mocatsoarm','motlboarm','mofpaarm','moaloarm']
algorithmNameList = ['mocsoarm','mopso','nsgaii','hmofaarm','mosaarm','mowsaarm','mocatsoarm','motlboarm','mofpaarm','moaloarm',
                     'modaarm','mohsbotsarm','modearm','nshsdearm','mogeaarm','mogsaarm','mossoarm','mowoaarm','mososarm',
                     'mocssarm','custom']
p = '../Experiments/RISK/'
if (not path.exists(p)):
    mkdir(p)
perf = Performances(algorithmNameList,criterionList,objectiveNames)
#d = Data(artificial=True)
d = Data('Data/Transform/risk.csv',header=0,indexCol=0)
# d.TransformToHorizontalBinary()
# d.Save('Data/Transform/water-treatment.csv')
d.ToNumpy()


# E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,iterationInitiale,sizeHead=sizeHead,path=p,display=False)
# E.Run()

g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/LeaderBoard/')
g.GraphExperimentation(algorithmNameList,'../Experiments/RISK/','LeaderBoard',nbIteration)
g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/Coverages/',display=True,save=True)
g.GraphAverageCoverages('../Experiments/RISK/',algorithmNameList)
g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/Distances/',display=True,save=True)
g.GraphAverageDistances('../Experiments/RISK/',algorithmNameList)
g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/ExecutionTime/',display=True,save=True)
g.GraphAverageExecutionTime('../Experiments/RISK/',algorithmNameList,nbIteration)












