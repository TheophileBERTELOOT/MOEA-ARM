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
parameterNames = ['mutationRate', 'crossOverRate']
p= 'HyperParameters/NSGAII/'
if(not path.exists(p)):
    mkdir(p)
d = Data('Data/Transform/flag.csv',header=0,indexCol=0)
d.ToNumpy()
alg = NSGAII(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
hyper = HyperParameters(parameterNames,sizeHead=10)
hyper.RandomSearch(100,alg,d.data)
hyper.SaveBestParameters(p+'bestParameters.json')
'''


nbIteration = 2
sizeHead = 10
iterationInitiale = 0
nbRepetition = 2 - iterationInitiale
populationSize = 100
objectiveNames = ['support','confidence','cosine']
criterionList = ['scores','execution time','distances','coverages']
algorithmNameList = ['custom']
algorithmNameList = ['mocsoarm','mopso','nsgaii','hmofaarm','mosaarm','mowsaarm','mocatsoarm','motlboarm','mofpaarm','moaloarm',
                     'modaarm','mohsbotsarm','modearm','nshsdearm','mogeaarm','mogsaarm','mossoarm','mowoaarm','mososarm',
                     'mocssarm','custom']
p = '../Experiments/TAE/'
if (not path.exists(p)):
    mkdir(p)
perf = Performances(algorithmNameList,criterionList,objectiveNames)
#d = Data(artificial=True)
d = Data('Data/Transform/tae.csv',header=0,indexCol=0)
# d.TransformToHorizontalBinary()
# d.Save('Data/Transform/water-treatment.csv')
d.ToNumpy()


E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,iterationInitiale,sizeHead=sizeHead,path=p,display=True)
E.Run()

g = Graphs(objectiveNames,[])
g.GraphExperimentation(algorithmNameList,'../Experiments/TAE/','LeaderBoard',nbIteration)












