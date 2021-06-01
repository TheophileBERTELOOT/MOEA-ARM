import pandas as pd
from src.Utils.Performances import *
from src.Utils.Data import *
from src.Utils.HyperParameters import *
from src.Utils.Experiments import *

'''
populationSize = 300
nbIteration = 10,
objectiveNames = ['support','confidence','klosgen']
parameterNames = ['s','a','c','f','e','w']
d = Data('Data/Transform/chess.data',header=0,indexCol=0)
d.ToNumpy()
modaarm = MODAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
hyper = HyperParameters(parameterNames)
hyper.RandomSearch(100,modaarm,d.data)
hyper.SaveBestParameters('HyperParameters/MODAARM/bestParameters.json')
'''

nbIteration = 10
nbRepetition = 5
populationSize = 100
objectiveNames = ['support','confidence','cosine']
criterionList = ['scores','execution time']
algorithmNameList = ['MOCSOARM','MOSSOARM']
algorithmNameList = ['mocsoarm','mopso','nsgaii','hmofaarm','mowsaarm','mocatsoarm','motlboarm',
                     'mofpaarm','moaloarm','modaarm','mohsbotsarm','modearm','nshsdearm','mogeaarm','mogsaarm','mossoarm']

perf = Performances(algorithmNameList,criterionList,objectiveNames)
#d = Data(artificial=True)
d = Data('Data/Transform/congress.csv',header=0,indexCol=0)
#d.TransformToHorizontalBinary()
#d.Save('Data/Transform/abalone.data')
d.ToNumpy()

E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,path='Experiments/Congress/')
E.Run()












