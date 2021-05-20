from src.Utils.Data import *
from src.Utils.Performances import *
import copy
import json

class HyperParameters:
    def __init__(self,nameList):
        self.hyperParameters  = {i : np.round(rd.random(),2) for i in nameList}
        self.bestParamters = {i : np.round(rd.random(),2) for i in nameList}

    def GetRandomParameters(self):
        for key,value in self.hyperParameters.items():
            self.hyperParameters[key] = np.round(rd.random(),2)

    def CheckIfPathExist(self,path):
        p = path.split('/')
        p = p[:-1]
        p = '/'.join(p)
        pathExist = os.path.exists(p)
        if not pathExist:
            os.mkdir(p)

    def RandomSearch(self,nbIter,alg,data):
        perf = Performances(['best','candidate'], ['scores'], alg.fitness.objectivesNames)
        algorithmName = 'best'
        for i in range(nbIter):
            self.GetRandomParameters()
            alg.ResetPopulation(data,self)
            for j in alg.nbIteration:
                alg.Run(data,j)
            alg.fitness.GetParetoFront()
            perf.UpdatePerformances(score=alg.fitness.paretoFront, i=j,
                                    algorithmName=algorithmName)
            algorithmName = 'candidate'
            if i >0:
                perf.UpdateLeaderBoard()
                best = perf.leaderBoard[0]
                candidate = perf.leaderBoard[1]
                if candidate>best:
                    candidateScores = perf.scores[perf.scores['algorithm'] == 'candidate']
                    candidateScores['algorithm'] = 'best'
                    perf.scores = copy.deepcopy(candidateScores)
                    self.bestParamters = copy.deepcopy(self.hyperParameters)
                    print(self.bestParamters)
                else:
                    perf.scores = perf.scores.drop(perf.scores[perf.scores['algorithm'] == 'candidate'].index)

    def SaveBestParameters(self,path):
        self.CheckIfPathExist(path)
        with open(path, "w") as outfile:
            json.dump(self.bestParamters, outfile)

    def LoadHyperParameters(self,path):
        with open(path) as f:
            self.hyperParameters = json.load(f)





