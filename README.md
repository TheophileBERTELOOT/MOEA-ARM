# MOEA-ARM
Multi Objectives Evolutionary Algorithm - Association rule mining
# Introduction
Gather a huge number of multi objectives evolutionary algorithm for solve the association rule mining problem. This library provide state-of-the-art algorithm and tools to compare algorithm performance, futhermore you can choose the metrics which be used for objectives. It's also possible to speed up algorithm by using GPUs.
# Experiments 
This library is initialy design to be used in experiments for articles, you have then the experiments module allowing you to specified a list of algorithm of criterion a number of repetition and it will perform the experiment and save the results somewhere:
```python
nbIteration = 20
nbRepetition = 5
populationSize = 200
objectiveNames = ['support','confidence','cosine']
criterionList = ['scores','execution time']
algorithmNameList = ['MOCSOARM','MOSSOARM']
perf = Performances(algorithmNameList,criterionList,objectiveNames)
d = Data('Data/Transform/tae.csv',header=None,indexCol=None)
d.ToNumpy()
E = Experiment(algorithmNameList,objectiveNames,criterionList,d.data,populationSize,nbIteration,nbRepetition,path='Experiments/TAE/',display=True)
E.Run()
```
# Hyperparameters managment 
It's possible to perform a RandomSearch on the hyperparameters of an algorithm and next save then load latter hyperparameters with the best performances. You can do this using this template :
```python
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
```
# Performance analysis tools
There is an example of instantiate the performance component.
```python
objectiveNames = ['support','confidence','klosgen']
criterionList = ['scores','execution time','distances','coverages']
algorithmNameList = ['CSOARM','MOPSO']
perf = Performances(algorithmNameList,criterionList,objectiveNames)
```
There is an example of update the performance component, usually in the main loop
```python
self.perf.UpdatePerformances(score=alg.fitness.paretoFront, executionTime=alg.executionTime, i=i,algorithmName=self.algListNames[k],coverage=alg.fitness.coverage,distance=alg.fitness.averageDistances)
```
  ## Domination scores graph
  ```python
graph = Graphs(objectiveNames,perf.scores,path='./Figures/Comparison/paretoFront'+str(i),display=False)
graph.GraphScores()
  ```
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/comparison.gif "Comparison of found Pareto front")
  ## Execution time graph
  Display the execution time for each iteration of each algorithm
  ```python
graph = Graphs(['execution Time'],perf.executionTime,path='./Figures/Comparison/execution_time')
graph.GraphExecutionTime()
  ```
   ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/ExecutionTime.png "Execution time")
 Display the average execution time for the full execution time, that mean the sum of the execution time of each iteration.
 ```python
 g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/ExecutionTime/',display=True,save=True)
 g.GraphAverageExecutionTime('../Experiments/RISK/',algorithmNameList,nbIteration)
 ```
 ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/ExecutionTimeAverage.png "Execution time")
  
  ## Number of Rules
  This graph allow to know how many rules each algorithm find in his pareto front. There is how compute and display the number of rules.
   ```python
graph = Graphs(objectiveNames, perf.nbRules, path='./Figures/Comparison/nbRules' + str(i), display=True)
graph.GraphNbRules()
  ```
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/nbRule.png "nbRules")
  
  ## Distances
  This graph display the average distance between one rule and all the others. The rules considered are the pareto front's. That allow us to know if the rules are diversified.
    ```python
    g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/Distances/',display=True,save=True)
    g.GraphAverageDistances('../Experiments/RISK/',algorithmNameList)
    ```
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/Distance.png "distance")
  
 ## Coverage
  This graph display the number of row of the dataset cover by the pareto front.
    ```python
    g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/Coverages/',display=True,save=True)
    g.GraphAverageCoverages('../Experiments/RISK/',algorithmNameList)
    ```
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/Coverage.png "distance")
  
  ## Fitness functions
  This graphs display the value of each fitness function for each iteration.
    ```python
    g = Graphs(objectiveNames,[],path='../Experiments/RISK/Graphs/LeaderBoard/')
    g.GraphExperimentation(algorithmNameList,'../Experiments/RISK/','LeaderBoard',nbIteration)
    ```
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/support.png "support")
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/confidence.png "confidence")
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/cosine.png "cosine")

# List of available algorithms
* NSGAII *Non-dominated Sorting Genetic Algorithm II* 
> DEB, Kalyanmoy, PRATAP, Amrit, AGARWAL, Sameer, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 2002, vol. 6, no 2, p. 182-197.
* MOWSAARM *MultiObjective Wolf Search Algorithm Association Rule Mining*
> AGBEHADJI, Israel Edem, FONG, Simon, et MILLHAM, Richard. Wolf search algorithm for numeric association rule mining. In : 2016 IEEE International Conference on Cloud Computing and Big Data Analysis (ICCCBDA). IEEE, 2016. p. 146-151.
* HMOFAARM *Hybrid MultiObjective Firefly Algorithm Association Rule Mining*
> WANG, Hui, WANG, Wenjun, CUI, Laizhong, et al. A hybrid multi-objective firefly algorithm for big data optimization. Applied Soft Computing, 2018, vol. 69, p. 806-815.
* MOCSOARM *MultiObjective Cockroach Swarm Optimization Association Rule Mining*
> KWIECIE??, Joanna et PASIEKA, Marek. Cockroach swarm optimization algorithm for travel planning. Entropy, 2017, vol. 19, no 5, p. 213.
* MOSAARM *MultiObjective Simulated Annealing Association Rule Mining*
> NASIRI, Mehdi, TAGHAVI, Leyla Sadat, et MINAEE, Behrouz. Multi-Objective Rule Mining Using Simulated Annealing Algorithm. J. Convergence Inf. Technol., 2010, vol. 5, no 1, p. 60-68.
* MOBARM *MultiObjective Bat Association Rule Mining*
> HERAGUEMI, Kamel Eddine, KAMEL, Nadjet, et DRIAS, Habiba. Multi-objective bat algorithm for mining interesting association rules. In : International Conference on Mining Intelligence and Knowledge Exploration. Springer, Cham, 2016. p. 13-23.
* MOPSO *MultiObjective Particle Swarm Optimization*
> COELLO, CA Coello et LECHUGA, Maximino Salazar. MOPSO: A proposal for multiple objective particle swarm optimization. In : Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No. 02TH8600). IEEE, 2002. p. 1051-1056.
* MOCatSOARM *MultiObjective Cat Swarm Optimization Association Rule Mining*
> BAHRAMI, Mahdi, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Cat swarm optimization (CSO) algorithm. In : Advanced optimization by nature-inspired algorithms. Springer, Singapore, 2018. p. 9-18.
* MOTLBOARM *MultiObjective Teaching learning Based Optimization Association Rule Mining*
> SARZAEIM, Parisa, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Teaching-learning-based optimization (TLBO) algorithm. In : Advanced optimization by nature-inspired algorithms. Springer, Singapore, 2018. p. 51-58.
* MOFPAARM *MultiObjective Flower Pollination Algorithm Association Rule Mining*
> AZAD, Marzie, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Flower pollination algorithm (FPA). In : Advanced optimization by nature-inspired algorithms. Springer, Singapore, 2018. p. 59-67.
* MOALOARM *MultiObjective Ant Lion Optimization Association Rule Mining*
> MANI, Melika, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Ant lion optimizer (ALO) algorithm. In : Advanced optimization by nature-inspired algorithms. Springer, Singapore, 2018. p. 105-116.
* MODAARM *MultiObjective Dragonfly Algorithm Association Rule Mining*
> ZOLGHADR-ASLI, Babak, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Dragonfly Algorithm (DA). In : Advanced Optimization by Nature-Inspired Algorithms. Springer, Singapore, 2018. p. 151-159.
* MOHBSOTSARM *MultiObjective Hybrid Bee Swarm Optimization Tabu Search Association Rule Mining*
>DJENOURI, Youcef, HABBAS, Zineb, DJENOURI, Djamel, et al. Diversification heuristics in bees swarm optimization for association rules mining. In : Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Cham, 2017. p. 68-78.
* MODEARM *MultiObjective Differential Evolution Association Rule Mining*
>ALATAS, Bilal, AKIN, Erhan, et KARCI, Ali. MODENAR: Multi-objective differential evolution algorithm for mining numeric association rules. Applied Soft Computing, 2008, vol. 8, no 1, p. 646-656.
* NSHSDEARM *Non-Dominated Sorting Harmony Search Differential Evolution Association Rule Mining*
>YAZDI, Jafar, CHOI, Young Hwan, et KIM, Joong Hoon. Non-dominated sorting harmony search differential evolution (NS-HS-DE): A hybrid algorithm for multi-objective design of water distribution networks. Water, 2017, vol. 9, no 8, p. 587.
* MOGEAARM *MultiObjective Gradient Evolution Algorithm Association Rule Mining*
>ABDI-DEHKORDI, Mehri, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Gradient Evolution (GE) Algorithm. In : Advanced Optimization by Nature-Inspired Algorithms. Springer, Singapore, 2018. p. 117-130.
* MOGSAARM *MultiObjective Gravitational Search Algorithm Association Rule Mining*
> RASHEDI, Esmat, NEZAMABADI-POUR, Hossein, et SARYAZDI, Saeid. GSA: a gravitational search algorithm. Information sciences, 2009, vol. 179, no 13, p. 2232-2248.
* MOSSOARM *MultiObjective Social-Spider Optimization Association Rule Mining*
>CUEVAS, Erik, CIENFUEGOS, Miguel, ZALD??VAR, Daniel, et al. A swarm optimization algorithm inspired in the behavior of the social-spider. Expert Systems with Applications, 2013, vol. 40, no 16, p. 6374-6384.
* MOWOAARM *MultiObjective Whale Optimization Algorithm Association Rule Mining*
>MIRJALILI, Seyedali et LEWIS, Andrew. The whale optimization algorithm. Advances in engineering software, 2016, vol. 95, p. 51-67.
* MOSOSARM *MultiObjective Symbiotic Organisms Search Association Rule Mining*
>CHENG, Min-Yuan et PRAYOGO, Doddy. Symbiotic organisms search: a new metaheuristic optimization algorithm. Computers & Structures, 2014, vol. 139, p. 98-112.
* MOCSSARM *MultiObjective Charged System Search Association Rule Mining*
>KAVEH, A. et TALATAHARI, Siamak. A novel heuristic optimization method: charged system search. Acta Mechanica, 2010, vol. 213, no 3, p. 267-289.

