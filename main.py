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



nbIteration = 50
sizeHead = 10
isPolypharmacy = False
iterationInitiale =0
nbRepetition = 1 - iterationInitiale
populationSize = 100
objectiveNames = ['support','confidence','cosine']
criterionList = ['scores','execution time','distances','coverages','tested']
algorithmNameList = ['mocsoarm','mopso','nsgaii','hmofaarm','mosaarm','mowsaarm','mocatsoarm','motlboarm','mofpaarm','moaloarm',
                     'modaarm','mohsbotsarm','modearm','nshsdearm','mogeaarm','mogsaarm','mossoarm','mowoaarm','mososarm',
                     'mocssarm','custom']
# algorithmNameList = ['custom']
updated = False
nameDataset = 'MUSHROOM'
p = '../Experiments/'+nameDataset+'/'
if (not path.exists(p)):
    mkdir(p)
# perf = Performances(algorithmNameList,criterionList,objectiveNames)
#d = Data(artificial=True)
d = Data('Data/Transform/wine.csv',header=0,separator=',')
# d = Data('Data/Transform/iris.csv')
#d.TransformToHorizontalBinary()
#d.Save('Data/Transform/wine.csv')
d.ToNumpy()
# g = Graphs(objectiveNames,d.data)
# g.dataTSNE()
#
# E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,iterationInitiale,sizeHead=sizeHead,path=p,display=False,update=updated,isPolypharmacy = isPolypharmacy)
# E.Run()
# #
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/')
# g.GraphSCCVsNBRules(algorithmNameList,'../Experiments/'+nameDataset+'/','nbRulesvsSCC',nbIteration)
#
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/')
# g.GraphSCCVsCoverage(algorithmNameList,'../Experiments/'+nameDataset+'/','coveragesvsSCC',nbIteration)
#
g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/LeaderBoard/')
g.GraphExperimentation(algorithmNameList,'../Experiments/'+nameDataset+'/','LeaderBoard',nbIteration)
#
#
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/')
# g.GraphNBRulesVsCoverages(algorithmNameList,'../Experiments/'+nameDataset+'/','nbRulesVsCoverage',nbIteration)
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/Coverages/',display=True,save=True)
# g.GraphAverageCoverages('../Experiments/'+nameDataset+'/',algorithmNameList,nbIteration)
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/Distances/',display=True,save=True)
# g.GraphAverageDistances('../Experiments/'+nameDataset+'/',algorithmNameList,nbIteration)
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/ExecutionTime/',display=True,save=True)
# g.GraphAverageExecutionTime('../Experiments/'+nameDataset+'/',algorithmNameList,nbIteration)
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/NbRules/',display=True,save=True)
# g.GraphAverageNBRules('../Experiments/'+nameDataset+'/',algorithmNameList,nbIteration)
# #
# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/')
# g.dataTSNEFromFileWithoutPareto(d.data)


# g = Graphs(objectiveNames,[],path='../Experiments/'+nameDataset+'/Graphs/')
# g.getAverage()

# perf = Performances(algorithmNameList, criterionList, objectiveNames)
# perf.CalculFitness('../Experiments/',nbIteration)
# g = Graphs(objectiveNames,[],path='../Experiments/fitness',display=True,save=True)
# g.GraphFitness('../Experiments/fitness.csv')


# perf = Performances(algorithmNameList, criterionList, objectiveNames)
# perf.CalculPerf('../Experiments/',nbIteration)

# perf = pd.read_csv('../Experiments/results.csv',header=0,index_col=0)
# print(perf[perf['dataset']=='MUSHROOM'][['suppCatchup',
#                 'confCatchup','cosCatchup']].sort_values(['algorithm']))












