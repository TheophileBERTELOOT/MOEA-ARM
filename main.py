import pandas as pd
from src.Utils.Performances import *
from src.Utils.Data import *
from src.Utils.HyperParameters import *
from src.Utils.Experiments import *
from os import path,mkdir
'''
populationSize = 200
nbIteration = 10
objectiveNames = ['support','confidence','cosine']
parameterNames = ['F','Fw','PAR']
p= 'HyperParameters/NSHSDEARM/'
if(not path.exists(p)):
    mkdir(p)
d = Data('Data/Transform/iris.csv',header=0,indexCol=0)
d.ToNumpy()
alg = NSHSDEARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
hyper = HyperParameters(parameterNames)
hyper.RandomSearch(50,alg,d.data)
hyper.SaveBestParameters(p+'bestParameters.json')
'''
'''
nbIteration = 20
nbRepetition = 5
populationSize = 200
objectiveNames = ['support','confidence','jaccard']
criterionList = ['scores','execution time']
algorithmNameList = ['mososarm','mocssarm']
#algorithmNameList = ['mocsoarm','mopso','nsgaii','hmofaarm','mowsaarm','mocatsoarm','motlboarm',
#                     'mofpaarm','moaloarm','modaarm','mohsbotsarm','modearm','nshsdearm','mogeaarm','mogsaarm','mossoarm']

perf = Performances(algorithmNameList,criterionList,objectiveNames)
#d = Data(artificial=True)
d = Data('Data/Transform/congress.csv',header=0,indexCol=0)
# d.TransformToHorizontalBinary()
# d.Save('Data/Transform/water-treatment.csv')
d.ToNumpy()

E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,path='Experiments/TAE/',display=True)
E.Run()
'''











