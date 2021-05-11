# MOEA-ARM
Multi Objectives Evolutionary Algorithm - Association rules mining
# Introduction
Gather a huge number of multi objectives evolutionary algorithm for solve the association rules mining problem. This library provide state-of-the-art algorithm and tools to compare algorithm performance, futhermore you can choose the metrics which be used for objectives. It's also possible to speed up algorithm by using GPUs.
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
 
# List of available algorithms
* NSGAII *Non-dominated Sorting Genetic Algorithm II* 
> DEB, Kalyanmoy, PRATAP, Amrit, AGARWAL, Sameer, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 2002, vol. 6, no 2, p. 182-197.
* MOWSAARM *MultiObjective Wolf Search Algorithm Association Rules Mining*
> AGBEHADJI, Israel Edem, FONG, Simon, et MILLHAM, Richard. Wolf search algorithm for numeric association rule mining. In : 2016 IEEE International Conference on Cloud Computing and Big Data Analysis (ICCCBDA). IEEE, 2016. p. 146-151.
* HMOFAARM *Hybrid MultiObjective Firefly Algorithm Association Rules Mining*
> WANG, Hui, WANG, Wenjun, CUI, Laizhong, et al. A hybrid multi-objective firefly algorithm for big data optimization. Applied Soft Computing, 2018, vol. 69, p. 806-815.
* MOCSOARM *MultiObjective Cockroach Swarm Optimization Association Rules Mining*
> KWIECIEŃ, Joanna et PASIEKA, Marek. Cockroach swarm optimization algorithm for travel planning. Entropy, 2017, vol. 19, no 5, p. 213.
* MOSAARM *MultiObjective Simulated Annealing Association Rules Mining*
> NASIRI, Mehdi, TAGHAVI, Leyla Sadat, et MINAEE, Behrouz. Multi-Objective Rule Mining Using Simulated Annealing Algorithm. J. Convergence Inf. Technol., 2010, vol. 5, no 1, p. 60-68.
* MOBARM *MultiObjective Bat Association Rules Mining*
> HERAGUEMI, Kamel Eddine, KAMEL, Nadjet, et DRIAS, Habiba. Multi-objective bat algorithm for mining interesting association rules. In : International Conference on Mining Intelligence and Knowledge Exploration. Springer, Cham, 2016. p. 13-23.
* MOPSO *MultiObjective Particle Swarm Optimization*
> COELLO, CA Coello et LECHUGA, Maximino Salazar. MOPSO: A proposal for multiple objective particle swarm optimization. In : Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No. 02TH8600). IEEE, 2002. p. 1051-1056.
* MOCatSOARM *MultiObjective Cat Swarm Optimization Association Rules Mining*
> BAHRAMI, Mahdi, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Cat swarm optimization (CSO) algorithm. In : Advanced optimization by nature-inspired algorithms. Springer, Singapore, 2018. p. 9-18.
* MOTLBOARM *MultiObjective Teaching learning Based Optimization Association Rules Mining*
> SARZAEIM, Parisa, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Teaching-learning-based optimization (TLBO) algorithm. In : Advanced optimization by nature-inspired algorithms. Springer, Singapore, 2018. p. 51-58.
* MOFPAARM *MultiObjective Flower Pollination Algorithm Association Rules Mining*
> AZAD, Marzie, BOZORG-HADDAD, Omid, et CHU, Xuefeng. Flower pollination algorithm (FPA). In : Advanced optimization by nature-inspired algorithms. Springer, Singapore, 2018. p. 59-67.

 
