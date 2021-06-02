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
d = Data('Data/Transform/chess.csv',header=0,indexCol=0)
d.ToNumpy()
modaarm = MODAARM(d.data.shape[1],populationSize,nbIteration,len(objectiveNames),objectiveNames,d.data)
hyper = HyperParameters(parameterNames)
hyper.RandomSearch(100,modaarm,d.data)
hyper.SaveBestParameters('HyperParameters/MODAARM/bestParameters.json')
'''

nbIteration = 20
nbRepetition = 5
populationSize = 200
objectiveNames = ['support','confidence','cosine']
criterionList = ['scores','execution time']
algorithmNameList = ['MOCSOARM','MOSSOARM']
algorithmNameList = ['mocsoarm','mopso','nsgaii','hmofaarm','mowsaarm','mocatsoarm','motlboarm',
                     'mofpaarm','moaloarm','modaarm','mohsbotsarm','modearm','nshsdearm','mogeaarm','mogsaarm','mossoarm']

perf = Performances(algorithmNameList,criterionList,objectiveNames)
#d = Data(artificial=True)
d = Data('Data/Transform/tae.csv',header=None,indexCol=None)
# d.TransformToHorizontalBinary()
# d.Save('Data/Transform/water-treatment.csv')
d.ToNumpy()

E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,path='Experiments/TAE/',display=True)
E.Run()












