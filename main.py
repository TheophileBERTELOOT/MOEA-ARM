import pandas as pd
from src.Utils.Performances import *
from src.Utils.Data import *
from src.Utils.HyperParameters import *
from src.Utils.Graphs import *
from src.Utils.Experiments import *
from os import path,mkdir
import sys


# g = Graphs([],[],path='Data/columnsRow',display=True,save=True)
# g.DatasetColumnsRows('Data/columnsRows.csv')

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

#refaire les 10 premiers dataset avec sso

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
nameDataset = 'MAGIC'
p = '../Experiments/'+nameDataset+'/'
if (not path.exists(p)):
    mkdir(p)
perf = Performances(algorithmNameList,criterionList,objectiveNames)
#d = Data(artificial=True)
d = Data('Data/Transform/magic.csv',header=0,indexCol=0,separator=',')
#d.TransformToHorizontalBinary()
#d.Save('Data/Transform/wine.csv')
d.ToNumpy()


E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,iterationInitiale,sizeHead=sizeHead,path=p,display=False)
E.Run()

g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/LeaderBoard/')
g.GraphExperimentation(algorithmNameList,'../Experiments/'+nameDataset+'/','LeaderBoard',nbIteration)
g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/Coverages/',display=False,save=False)
g.GraphAverageCoverages('../Experiments/'+nameDataset+'/',algorithmNameList)
g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/Distances/',display=False,save=False)
g.GraphAverageDistances('../Experiments/'+nameDataset+'/',algorithmNameList)
g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/ExecutionTime/',display=False,save=False)
g.GraphAverageExecutionTime('../Experiments/'+nameDataset+'/',algorithmNameList,nbIteration)
g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/NbRules/',display=False,save=False)
g.GraphAverageNBRules('../Experiments/'+nameDataset+'/',algorithmNameList,nbIteration)













