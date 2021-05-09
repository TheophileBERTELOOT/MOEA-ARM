from src.Algorithm.MOPSO import *
from src.Algorithm.MOBARM import *
from src.Algorithm.NSGAII import *
from src.Algorithm.CSOARM import *
from src.Algorithm.HMOFAARM import *
from src.Algorithm.MOSAARM import *
from src.Algorithm.MOWSAARM import *
from src.Algorithm.MOCatSOARM import *

data = pd.read_csv('Data/bankrupt.csv')
data = data.to_numpy()
#alg = MOPSO(data.shape[1],200,20,3,['support','confidence','lift'])
#alg = MOBARM(data.shape[1],50,20,10,3,['support','confidence','lift'],save=True,display=True,path='Figures/MOBARM/')
#alg = NSGAII(data.shape[1],200,20,3,['support','confidence','lift'],)
#alg = CSOARM(data.shape[1],200,20,3,['support','confidence','lift'])
#alg = HMOFAARM(data.shape[1],100,20,3,['support','confidence','lift'])
#alg = MOSAARM(data.shape[1],10,30,2,['support','confidence','lift'],save=True,display=True,path='Figures/MOSAARM/')
#alg = MOWSAARM(data.shape[1],200,20,3,['support','confidence','lift'],save=True,display=True,path='Figures/MOWSAARM/')
alg = MOCatSOARM(data.shape[1],200,20,3,['support','confidence','lift'],save=True,display=True,path='Figures/MOCatSOARM/')

alg.Run(data)