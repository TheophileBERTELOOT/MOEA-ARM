# MOEA-ARM
Multi Objectives Evolutionary Algorithm - Association rule mining
# Introduction
Gather a huge number of multi objectives evolutionary algorithm for solve the association rule mining problem. This library provide state-of-the-art algorithm and tools to compare algorithm performance, futhermore you can choose the metrics which be used for objectives. It's also possible to speed up algorithm by using GPUs.
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
criterionList = ['scores','execution time']
algorithmNameList = ['CSOARM','MOPSO']
perf = Performances(algorithmNameList,criterionList,objectiveNames)
```
There is an example of update the performance component, usually in the main loop
```python
perf.UpdatePerformances(score=alg.fitness.paretoFront,executionTime=alg.executionTime,i=i,algorithmName=algorithmNameList[k])
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
  ## LeaderBoard
  the leaderboard display a sorted list of the average number of dominated solution by each solution by algorithm.
  ```python
perf.UpdateLeaderBoard()
  ```
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/LeaderBoard.PNG "leaderboard")
  
  ## Number of Rules
  This graph allow to know how many rules each algorithm find in his pareto front. There is how compute and display the number of rules.
   ```python
graph = Graphs(objectiveNames, perf.nbRules, path='./Figures/Comparison/nbRules' + str(i), display=True)
graph.GraphNbRules()
  ```
  ![alt text](https://github.com/TheophileBERTELOOT/MOEA-ARM/blob/main/Figures/Readme/nbRules.png "nbRules")
 
# List of available algorithms
* NSGAII *Non-dominated Sorting Genetic Algorithm II* 
> DEB, Kalyanmoy, PRATAP, Amrit, AGARWAL, Sameer, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 2002, vol. 6, no 2, p. 182-197.
* MOWSAARM *MultiObjective Wolf Search Algorithm Association Rule Mining*
> AGBEHADJI, Israel Edem, FONG, Simon, et MILLHAM, Richard. Wolf search algorithm for numeric association rule mining. In : 2016 IEEE International Conference on Cloud Computing and Big Data Analysis (ICCCBDA). IEEE, 2016. p. 146-151.
* HMOFAARM *Hybrid MultiObjective Firefly Algorithm Association Rule Mining*
> WANG, Hui, WANG, Wenjun, CUI, Laizhong, et al. A hybrid multi-objective firefly algorithm for big data optimization. Applied Soft Computing, 2018, vol. 69, p. 806-815.
* MOCSOARM *MultiObjective Cockroach Swarm Optimization Association Rule Mining*
> KWIECIEŃ, Joanna et PASIEKA, Marek. Cockroach swarm optimization algorithm for travel planning. Entropy, 2017, vol. 19, no 5, p. 213.
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
*MOGEAARM *MultiObjective Gradient Evolution Algorithm Association Rule Mining*
>ABDI-DEHKORDI, Mehri, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Gradient Evolution (GE) Algorithm. In : Advanced Optimization by Nature-Inspired Algorithms. Springer, Singapore, 2018. p. 117-130.
