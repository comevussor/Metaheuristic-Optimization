# Metaheuristic-Optimization
8 problems to optimize with Python
For TSP problems, data is downloaded from the web.
For functions problems, algorithms need the file [data.py](data.py) where the given functions shifts are stored.  

- [Travelling Salesman Problem (TSP) with `n = 38` cities in Djibouti](#travelling-salesman-problem--tsp--with--n---38--cities-in-djibouti)
  * [Implementation of a genetic algorithm on TSP problem for Djibouti `n = 38` (target is `6,656`)](#implementation-of-a-genetic-algorithm-on-tsp-problem-for-djibouti--n---38---target-is--6-656--)
    + [Improvement tries](#improvement-tries)
  * [Implementation of closest neighbour algorithm on TSP problem for Djibouti `n = 38` (target is `6,656`)](#implementation-of-closest-neighbour-algorithm-on-tsp-problem-for-djibouti--n---38---target-is--6-656--)
- [Travelling Salesman Problem (TSP) with `n = 194` cities in Quatar](#travelling-salesman-problem--tsp--with--n---194--cities-in-quatar)
  * [Implementation of a genetic algorithm on TSP problem for Quatar `n = 194` (target is `9,352`)](#implementation-of-a-genetic-algorithm-on-tsp-problem-for-quatar--n---194---target-is--9-352--)
  * [Implementation of closest neighbour algorithm on TSP problem for Quatar](#implementation-of-closest-neighbour-algorithm-on-tsp-problem-for-quatar)
  * [Conclusion regarding TSP problem with genetic algorithm and best neighbour](#conclusion-regarding-tsp-problem-with-genetic-algorithm-and-best-neighbour)
- [Optimize shifted sphere function in dimension `d = 50` and `d = 500`](#optimize-shifted-sphere-function-in-dimension--d---50--and--d---500-)
  * [Gradient descent algorithm](#gradient-descent-algorithm)
    + [Gradient descent with random data](#gradient-descent-with-random-data)
    + [Gradient descent with real data](#gradient-descent-with-real-data)
  * [Particle sworm optimization](#particle-sworm-optimization)
  * [Conclusion regarding shifted sphere with gradient descent and particle sworm.](#conclusion-regarding-shifted-sphere-with-gradient-descent-and-particle-sworm)
- [Optimize shifted Schwefel's problem 2.21 in dimension `d = 50` and `d = 500`](#optimize-shifted-schwefel-s-problem-221-in-dimension--d---50--and--d---500-)
  * [Gradient descent algorithm for unknown shift](#gradient-descent-algorithm-for-unknown-shift)
  * [Particle sworm optimization](#particle-sworm-optimization-1)
  * [Conclusion regarding shifted Schweffel's problem with gradient descent and particle sworm.](#conclusion-regarding-shifted-schweffel-s-problem-with-gradient-descent-and-particle-sworm)
- [Optimize shifted Rosenbrock's function with simulated annealing coupled with gradient method](#optimize-shifted-rosenbrock-s-function-with-simulated-annealing-coupled-with-gradient-method)
- [Optimize shifted Rastrigin's function with simulated annealing algorithm, coupled with gradient local search](#optimize-shifted-rastrigin-s-function-with-simulated-annealing-algorithm--coupled-with-gradient-local-search)
  * [Simulated annealing coupled with gradient descent in dimension `50`](#simulated-annealing-coupled-with-gradient-descent-in-dimension--50-)
  * [Implement with real data in dimension `50`](#implement-with-real-data-in-dimension--50-)
  * [Extend the method to dimension `500`](#extend-the-method-to-dimension--500-)
    + [barycentric approach in dimension `500` with real data](#barycentric-approach-in-dimension--500--with-real-data)
  * [Comparison between dimension `50` and dimension `500`](#comparison-between-dimension--50--and-dimension--500-)
- [Optimize shifted Griewank’s Function in dimension `50` and `500`](#optimize-shifted-griewank-s-function-in-dimension--50--and--500-)
  * [Optimize shifted Griewank’s Function in dimension `50` with gradient descent](#optimize-shifted-griewank-s-function-in-dimension--50--with-gradient-descent)
  * [Optimize shifted Griewank’s Function in dimension `500` with gradient descent](#optimize-shifted-griewank-s-function-in-dimension--500--with-gradient-descent)

# Travelling Salesman Problem (TSP) with `n = 38` cities in Djibouti

Data source : [National Traveling Salesman Problem, University of Waterloo (Canada)](http://www.math.uwaterloo.ca/tsp/world/dj38.tsp)

[OPEN MAIN CODE : TSP38.py](TSP38.py)  
[OPEN UTILITIES : TSP_util.py](TSP_util.py)

We are looking for the shortest path so as to visit all cities without visiting twice the same.  
We know the coordinates of each city on a 2D euclidian plane.  
We first build a distance matrix, using usual 2-norm. This is a symmetric matrix with null diagonal and strictly positive values everywhere else satisfying the triangular inequality. The computation is asymptotically of order `n^2`.

![](images/TSP38_distances.png)

A solution to this problem is a specific arrangement of `[1, ..., n]` . It can be viewed as a hollow `n*n` matrix where `x_(i,j) = 1` whenever the path includes going from city `i` to city `j` . It has only `n` non-zero values (return to origin) and it is of rank `n`. The total distance function to minimize is the sum of the product of the distance matrix by the solution matrix. Obviously, this is a very simple product and it can be simplified summing elements read one by one in the distance matrix. It goes from an order `n^3` to an order `n` algorithm. Together with the generation of random solution, it is the main purpose of the [`TSP` class](https://github.com/jMetal/jMetalPy/blob/master/jmetal/problem/singleobjective/tsp.py) proposed in `jmetalpy`. We use it. Note that it also suggests a method to compute the distance matrix but it is far from being optimized compared to `scipy` method which uses a general optimized approach with numpy arrays [Minkowski distance](https://github.com/scipy/scipy/blob/v1.4.1/scipy/spatial/kdtree.py#L15-L55). We therefore create a new class named `TSP2` to adjust to our local problem. It will be of no consequence for using  `jmetalpy` later on because we keep all attributes and public methods.

It is a discrete non-linear problem and we can say that one solution is related to another one by a series of transpositions (any permuation can be decomposed in transpositions). This kind of modification is strongly relevant in a genetic context. It seems relevant to try and use a genetic algorithm in this context. We must be careful with this method because it may not converge and indeed, [as shown in 2005 by H. Abdulkarim and I.F. Alshammari](https://www.researchgate.net/publication/280597707_Comparison_of_Algorithms_for_Solving_Traveling_Salesman_Problem), it happens that it does not converge in the TSP case. We could also tried Simulated Annealing, Tabu Search, Particle Sworm Optimization that fits we the discrete non-linear problems. But we have found in litterature that in TSP problem, there are many local optima and genetic algorithm is known to be indifferent to these issues. For instance, [in 2012 W. Hui showed](https://www.sciencedirect.com/science/article/pii/S2211381911002232) (note that these are a conference proceedings and english syntax is very poor) that for `n = 51` we can get an excellent result with an ant colony algorithm but it is very sensitive to parameters. As I am not well versed in this subject, I'd rather use a more robust method. 

Note that we also had to re-write the `PermutationSwapMutation` of `jmetalpy` as it contains an error in the available package (see details in the code).

## Implementation of a genetic algorithm on TSP problem for Djibouti `n = 38` (target is `6,656`)

Let `D` be the distance matrix.

The number of possible paths of the travelling salesman is of order `( n - 1 )!` which in our case amounts to `10^34` which is huge. On the other hand, the length of the path is bounded between `n * min( D+ )` and `n * max( D+ )` where `D+` is the set of strictly positive values of `D`. In our case, the bounds are roughly `200` and `70,000` . It is necessary to choose a population size to feed the genetic algorithm that is significant with respect to the size of our problem but keeping in mind that the complexity of the algorithm is directly proportional to the population size. We choose `n_0 = 1,000` with a maximum number of iterations at `20,000` this large population is likely to favor diversity of our population. We keep in mind that in Hui computation, the optimal result has been found after less than `400` iterations for `n = 51` even if the effective stopping criterion has been the number of iterations (`2,000`). We could refine our model choosing a stopping criteriONa as a number of iterations without improvement.

Regarding mutation procedure, the minimum change that can be introduced in a solution is a transposition. We can either choose the permuation randomly or optimize this process (which introduces an overhead). We choose it randomly to begin with, that will save computation. We take Hui's probability at `0.01` .

Regarding crossover, we keep the classical method in litterature. We swap 2 sections among parents (swap genes of random size) and replace remaining alleles in other slots according to original sequential order as for instance :

      P1 = (1,  2,3,4,5,  6,7) ; P2 = (4,  3,1,2,7,  6,5)

      swap from index 2 to 5 :
      S1 = (1,  3,1,2,7,  6,7) ; S2 = (4,  2,3,4,5,  7,5)

      re-arrange :
      S1 = (4,  3,1,2,7,  5,6) ; S2 = (1,  2,3,4,5,  7,6)

We also take Hui's probability at `0.3` .

We use a roulette wheel selection that sets the probability to mutate or mate inversely proportional to the length of the trajectory. Probability is proportionnal to the trajectory length. Note that the `RouletteWheelSelection` class proposed in `jmetalpy` has to be reworked as it is suited only for maximization problem (see details in the code).

We set random seed at 1203 for reproducibility.  
Based on these results, we get a first result at `15,976`, to be compared with the target `6,656`.

### Improvement tries

We first increase the selection severity taking a probability depending exponentially on the trajectory length. The result is not improved at `17,983` . It shows that intensification does not improve the result. We should rather try to diversity.  
Now, instead of choosing a random mutation, we choose the mutation of best improvement. It is a bit costly in terms of computing (order `n` with optimized algorithm) but it will give a path that may be further from the parent. Score is `12,760`, this is a real improvement, we keep the optimization.  
If we cumulate both changes, we get a fitness at `14,224`. There is no improvement.

We change our strategy. We know from Dr Nakib that genetic algorithm might perform well with small population. Therefore we leave Hui's strategy to have a large population and reduce our population down to `100` . Number of evaluations is proportionally increased up to `100,000`. Fitness is `10,694` . This is a significant improvement.  
We try to work on this approach. Considering the population decrease, we have set a mutation probability that induces a number of mutation at each generation close to `0` . This is not good for diversity. We change its order of magnitude up to `0.5` . New fitness is `8,091` which is again a significant improvement. Computation takes `22 sec` on my computer.  

At this point, we re-run our programm with another seed (`1,815`) to check that improvements are not specific to our case. Things are OK, fitnesses follow the same evolution. We come back to our initial seed.

Computation takes around `22 sec`.

Considering the high variability of genetic algorithm, we perform the computation `100` times and take the minimum result, that is `7,025`, to be compared with the targe `6,656`. We stop our quest at this point. This is very costly (around `40 min`) even if it could easily be parallelized.

## Implementation of closest neighbour algorithm on TSP problem for Djibouti `n = 38` (target is `6,656`)

This is a deterministic algorithm known to be quite efficient as a first approach. We initiate the algorithm from each node and find the best path from this node. The best of the `38` paths has length `6,770`. However, since `n` is small, we can optimize those paths by a sequence of best transpositions. In that case, the best path has length `6,659` which is a very good performance. We keep it. Computation time is less than `1 sec`.

Eventually, this last method is more efficient than genetic algorithm but with almost no chance to reach the best path. Genetic algorithm is less efficient on average but gives us a chance to reach the best path.

# Travelling Salesman Problem (TSP) with `n = 194` cities in Quatar

Data source : [National Traveling Salesman Problem, University of Waterloo (Canada)](http://www.math.uwaterloo.ca/tsp/world/qa194.tsp)
[OPEN MAIN CODE : TSP38.py](TSP194.py)  
[OPEN UTILITIES : TSP_util.py](TSP_util.py)

## Implementation of a genetic algorithm on TSP problem for Quatar `n = 194` (target is `9,352`)

The major issue of this new problem is the increase of `n` and consequently the high computing demand. In this case, the target is `9,352` with rough bounds for our trajectories `193` and `281,600` .

![](images/TSP194_distances.png)

Seed is set at `1,815` for reproducibility.

If we use the same parameters as in previous case, we get :

| Population size | Mutation probability | Crossover probability | Roulette wheel selection | Max evaluations | Fitness |
| -- | -- | -- | -- | -- | -- |
| 1,000 | 0.01 | 0.3 | inverse proportional | 20,000 | 71,03 |
| 1,000 | 0.01 | 0.3 | exponential (0.001)  | 20,000 | 57,097 |
| 100 | 0.01 | 0.3 | inverse proportional | 100,000 | 24,065 |
| 100 | 0.5 | 0.3 | inverse proportional | 100,000 | 19,628 |
| 100 | 0.5 | 0.3 | exponential (0.000,1) | 100,000 | 21,825 |
| 30 | 10 | 0.5 | exponential (0.000,05) | 100,000 | 20,675 |

Note that last computations takes around `100 sec` on my computer. I did implement the optimal mutation but the computation demand is high. I also noted that we get similar results with population of 30 and offspring of 10 individuals.

We roughly have the same performance profile except that last result is quite far from optimal solution and we note that exponential method for roulette wheel selection might be efficient for large populations but not for small populations.  
This genetic algorithm is costly and no so efficient as far as we can see. 

## Implementation of closest neighbour algorithm on TSP problem for Quatar

 We try the closest neighbour path and we find a trajectory length equal to `11,330` which is quite good considering the target `9,352` and our previous result.

We can optimize this path with best transpositions to get it down to `10,877` by deterministic best mutations or when we randomize the choice of mutation among the best ones. It means that this point is quite attractive. Or worth if we completely randomize the mutation. We stop the process at this point.

## Conclusion regarding TSP problem with genetic algorithm and best neighbour

We see that a genetic algorithm does not require a large population to perform. This seems in contradiction with a lot of litterature and it is counter-intuitive. We also observed that it is difficult to predict the effect of selection process. However, it is clear that the use of an optimized mutation is very useful in both cases even if the computational cost can not be neglected. It could be interesting to study other methods like different types of crossover or different types of mutations or introducing new individuals every so many generation.

As a remark, we can also note that the package `jmetalpy` is in many aspects not optimized and may even contain small mistakes. Hovever, the code is simple enough to be proofread.

We also observed that the best neighbour algorithm is very efficient, especially if combined with a best transposition descent algorithm. Finding the best path from a given city is a computation of order `n` . Finding the best path among all cities is of order `n` therefore the algorithm is of order `n^2` . Finding the best mutation for a given path should be of order `n^2` (test each mutation among `n ( n - 1 )` ) but with numpy optimized arrays, it is possible to reduce it to get a power of `n` between `1` and `2` . I don't know about the length of the descent but it should somehow grow with `n` too.

# Optimize shifted sphere function in dimension `d = 50` and `d = 500`

[Description of the function : CEC2008_TechnicalReport.pdf](CEC2008_TechnicalReport.pdf)  
[Data source : data.py](data.py)  
[OPEN MAIN CODE : Function_sphere.py](Function_sphere.py)   
[OPEN UTILITIES : Functions_util.py](Functions_util.py)

For reproducibility, results are presented with `random.seed(2610)` and `rng = default_rng(1611)` .

The bias is set to `-450`, the shift may vary within the hypercube centered on the origin with side length `200` .  
This is a non-linear real continuous function. We can first use a gradient method with estimated gradient descent.

## Gradient descent algorithm

### Gradient descent with random data

We test our method `25 times` with random values of the shifts to be sure that our result does not depend on the specificity of the given data. Indeed, this method is deterministic and it does not make sense to test it several times on the same initial value.

Usually, the gradient method is used when we know the gradient function but it is possible to implement it with an estimated value of the gradient. The cost is of order `d` . Note that this algorithms can also be used for a local optimization after a global search with another method. For a real function, it seems important to me to try this method before anything.

Results for an estimated gradient descent with optimized step and adjusted learning rate (target value is `-450`) :

We estimate gradient with a minimal step : `eps = 10^-4`  
Learning rate is : `ro = 0.5` to move fast at the beginning, divided by `10` after `25 steps` and again after `50 steps`.  
Minimal step to stop is : `dimension * 10^-4` (minimal step is the euclidian norm of the move, scaling is necessary)

Fitnesses reached in dimension `50` :

  [-449.99997501 -449.99997501 -449.99997501 -449.99997501 -450.  
    -450.         -449.999975   -449.99997501 -449.99997501 -450.  
    -449.99999169 -449.99997501 -449.99997501 -450.         -449.99997501  
    -449.99997502 -449.99997502 -450.         -450.         -450.  
    -449.99997502 -449.99997501 -450.         -450.         -449.99999199]  

After number of iterations :

  [52. 52. 52. 52. 53. 55. 52. 52. 52. 55. 52. 52. 54. 53. 52. 52. 52. 53.
    53. 53. 52. 52. 55. 55. 56.]

Fitnesses reached in dimension `500` :

  [-449.99750203 -449.99750236 -449.99750373 -450.         -449.99750194  
    -450.         -450.         -450.         -449.99750203 -449.99750351  
    -450.         -449.99750427 -449.99750222 -449.99750202 -449.99755304  
    -449.99750208 -449.99750221 -450.         -449.99750425 -449.99750444  
    -450.         -449.99750216 -449.99750201 -449.99750389 -450.         ]  

After number of iterations :

  [26. 26. 26. 27. 26. 27. 27. 29. 26. 26. 29. 26. 26. 26. 28. 26. 26. 27.  
    26. 26. 27. 28. 26. 26. 27.]

As a global optimization, this gradient can be accepted only if we know at least that the function is unimodal.

### Gradient descent with real data

Computation takes a few seconds. Performances are the same.  
Errors with real shift are as expected :  
`8.799052636572125e-07` in dimension `50`  
`0.04997773768883722` in dimension `500`

## Particle sworm optimization

We know that the algorithm is very sensitive to parameters. We therefore make a quick trial with different learning rates.

![](images/sphere_PSO_dim50.png)

We observe that for this unimodal function, the highest is the global learning rate, the best is the result, independently of local learning rate. Conversely, in case of unknown function, this may be an indication that our function is most likely unimodal.

Based on this trial, in dimension `50`, we take global learning rate `g_rate = 5` and local learning rate `l_rate = 0.2` . PSO, unlike genetic algorithm, does not introduce much diversity, therefore we need a large initial population, set at `500` . For computational limitation reasons, we set `max_iter = 1,000` . Considering our window, the hypercube, we keep maximum velocity at `5 units/iteration` . We leave friction at default value `0.5` because we have observed fast convergence when testing learning rates.

One run takes around `10 sec` on my computer. Results are quite close to optimality, we are mostly within a few units of the minimum which is `-450` (`-449.82` in our case). Which is good considering the huge value range of the objective function.

In high dimension, checking constraints is a bit costly due to poor optimization of the available algorithm, that's why it is cheaper and as efficient to introduce a penalizing term. We introduce a penality that is not huge (not like a "wall") to avoid any unwanted side effect considering that somme coordinates of the shift (x target) might be close to the border.

Considering the high dimensionality, we will have to boost the positive effects of global learning rate because of the noise introduced by dimensionality. We make a test similar to the previous one.

![](images/sphere_PSO_dim500.png)

We observe the same global behaviour. Because of dimensionality, we also decide to increase number of iterations to `1,500` even if the cost is heavy considering a population of `1,000` . We take a global learning rate equal to `7`, keep a local learning rate at `0.2` . But because of dimensionality, we reduce maximum velocity down to `1` because the maximum velocity is per coordinate, therefore its effect is multiplied by dimensionality. In order to balance out this limitation, we slightly increase friction so that the sworm does not stop before the optimum, say `0.9` . Knowing that those parameters may quickly move an individual out of the domain, we take initial values in a slighty reduced hypercube to avoid useless search outside the domain.

The algorithm converges but `1,500` iterations is not enough to reach the target. However, as mentioned before, we can make a local search from this result with gradient descent algorithm. We get a fitness around `-140`. But time to reach those results is `700 sec` on my computer, that is 70 times more than is dimension `50` .

## Conclusion regarding shifted sphere with gradient descent and particle sworm.

We observe that gradient descent is thousand times more efficient than particle sworm in this case. But we must remember that gradient usually converges only locally to the nearest local optimum and that it may not converge at all. On the other hand, setting parameters for PSO algorithm is quite touchy and algorithm is heavy but it usually ensures a global search.

# Optimize shifted Schwefel's problem 2.21 in dimension `d = 50` and `d = 500`

[Description of the function : CEC2008_TechnicalReport.pdf](CEC2008_TechnicalReport.pdf)  
[Data source : data.py](data.py)  
[OPEN MAIN CODE : Function_Schweffel.py](Function_Schweffel.py)   
[OPEN UTILITIES : Functions_util.py](Functions_util.py)

Conditions and reproducibility seeds are the same as before.

Bias is `-450`, shift is in the centered hypercube of side `200`.

## Gradient descent algorithm for unknown shift

We must be careful because this is not a smooth function. Therefore we compute a pseudo-gradient that can take misleading values. We have a function that is continuous but only piecewise derivable. It can be interesting in this case to "jump" over those discontinuites : we would compute the pseudo gradient `f(x + eps/2) - f(x - eps/2)` with `eps` greater than usual. Instead of taking `eps << 1`, we take a value arount `1` . Obviously we loose a lot of precision in this process and it can be of interest to rescale `eps` along the process.

We can anticipate that the gradient algorithm will be less efficient than previously because it will move one coordinate after the other. In that case, it can be interesting to have a coefficient `ro` (the coefficient multiplying gradient in the descent step) that will be adjusted along the process : start with big leaps to move out of those "plains" and get more precise when approaching the target. That is why I decide to start with `ro = 7` during half of the iterations, `ro = 0.7` during a quarter of the iterations, `ro = 0.07` during a quarter of the iterations. Initial `ro` value is chosen as a parameter.  
I have also tried an algorithm with optimized descent, adjusting the leap length at each step but is gives no satisfactory result.

In dimension `50`, we keep a fixed small value for `eps`, with `1,000` iterations, we get a decent convergence with end values between `-449,9` and `-450` when the target is `-450` . The computation cost is low, taking aroung `1 sec` on my machine. Since we know that the algorithm converges we can run it longer with smaller values of `ro` to get more precision at low cost.

In dimension `500`, we take an initial value of `eps = 1` and rescale it twice is the same proportions as `ro` . With `3,000` iterations starting from `ro = 50` we get end values between `-449.6` and `-449.9` when the target is `-450`. Each run takes aroung `24 sec` on my machine. Because of dimension, function evaluation time is multiplied by `10` . As we know the function, we know that search will also be complexified.

Results are similar with random shifts and with real data.

Errors on the shift's estimates are respectively `0.056` and `0.965` which is quite good considering the dimensionality.

In this case, simplex method would probably be efficient but in order to compare with the previous case, I will keep PSO.

## Particle sworm optimization

We we test global and local learning rates, we observe a very different pattern from the previous function. We have an optimal zone along a line joining `(local rate = 2.5, global rate = 5.5)` with `(local rate = 5.5, global rate = 2.5)` , we choose `4` for both as a compromise.

![](images/Schweffel_PSO_dim50.png)

We can observe that in dimension `50`, the algorithm has a poor convergence. Even if we easily reach a fitness equal to `-400`, the end of the job is very costly. This is a typical case were it would be interesting to couple this algorithm with a local search with a gradient descent. PSO algorithm tells us that we are probably close to a global optimum. Then we switch to a deterministic algorithm to find the nearest local optimum, hoping that it will be global.

We take a population of `50`, limit our algorithm to `1,500` iterations and we get an average value with 25 trials around `-405` with a convergence in all cases. We can take the average values of limits as an initial vector for a gradient algorithm. We get a performance similar to the pure gradient algorithm (`-449,96`) but with a convergence after `1,000` iterations only. It is therefore interesting to couple global and local search algorithms.

In dimension `500` it is quite difficult to identify a clear optimal zone for local and global learning rates.

![](images/Schweffel_PSO_dim500.png)

In fact, after a few tries, we see that it is very difficult to get even a decent convergence. The algorithm gets stuck around a fitness equal to `-360` due to loss of diversity. I also observe that in spite of introducing a penalizing term to stay in the bounds, the "sworm" has a tendency to go out of control. If I try to apply the constraints directly, the computation overhead is too much due to poor optimization of the algorithm. Therefore I introduce a `callback` function to clip the vector in the domain. Convergence is much improved as far as the variable is concerned.

In order to keep diversity, I prefer running quickly (`2 to 3 sec` each) the algorithm `100` times to generate a new initial population to be used for a new run. I keep previous parameters for the first part. Regarding the second part, I reduce `max_velocity` down to `0.5` and take default value `0.5` for `friction`, as we should already be close to the solution. But unfortunately, it is not conclusive. The algorithm gets stuck very quickly around `-350`.

In dimension `500` it is therefore much more interesting to use only estimated gradient descent algorithm as PSO is way to heavy to handle. I get no interesting results below a population of `10,000` .

## Conclusion regarding shifted Schweffel's problem with gradient descent and particle sworm.

As a conclusion we observe that it may be interesting to couple gradient descent with PSO in dimension `50` but only the gradient descent gives good performance in dimension `500` . It would be necessary to run it from a large number of initial points to have a high probability to get a global optimum.

# Optimize shifted Rosenbrock's function with simulated annealing coupled with gradient method

[Description of the function : CEC2008_TechnicalReport.pdf](CEC2008_TechnicalReport.pdf)  
[Data source : data.py](data.py)  
[OPEN MAIN CODE : Function_Rosenbrock.py](Function_Rosenbrock.py)   
[OPEN UTILITIES : Functions_util.py](Functions_util.py)

Bias is `-390` and shift is in the centered hypercube of side `200`.

Rosenbrock function is multimodal, therefore it is useless to try gradient method before a global search.  
Moreover, when we compute the gradient of the function, we observe that in addition to the global optimum at `x = o`, there is an area where gradient is null for all coordinates except the first one. At this point, all coordinates of `x - o` are equal except the first one. This means that it is a very deep pit, all the more attractive that dimension is high. Indeed, we'll see that Rosenbrock function is easily optimized in dimension `50` but in dimension `500`, things are more difficult.

We are trying to minimize it with a personnalized version of simulated annealing. As there is too much dependency between parameters in usual implementations, I find it easier to work with less parameters : we decide a maximum number of iterations (`maxIter`), the initial step (`divers`), the final step  (`intens * divers`) and a parameter for acceptance (`accept`).  
Perturbation is a simple random vector in `[-1,1]`. We scale it with factor `divers * intens^(iter/maxIter)`.  
Probability of acceptance is `exp(- nrjVar * accept)` where `nrjVar` is a positive variation of energy.  
This is quite simple but turns out to be quite efficient. That way we don't need to define a temperature which is often redundant with iterations count.

In dimension `50` we set `maxIter = 500,000` to limit computation time.  
We set initial step at `divers = 0.3` and final step at `divers * intens = 0.000,3`, this is a reasonable tradeoff between the necessity to move in a range of `200` and to localize solution at a precision less than `0.1` . We keep `accept = 1` (Metropolis–Hastings setting). Each run takes around `20 sec`. We run the algorithm 5 times on random data. Best values are between `-335` and `-365` which is good to take to initiate a gradient. We keep those parameters to work on real data, we get same kind of results. But it is worth noticing that it is useful to run the algorithm several times as the variability is not negligible. We perform an optimized gradient method on the best result and we get a much better result within less than `10 sec` . We approach real solution by less than `0.2` on each coordinate and get a final Rosenbrock function value at `-389.99` for a target at `-390`. Error is `0.2` in euclidian norm : we found the right point.

In dimension `500` we have to significantly lengthen the flight at `maxIter = 3,000,000`, we keep initial step at `divers = 0.3` but due to dimensionality effects, we scale down the final step at `intens * divers = 0.000,03` . Obviously, this might become an issue at some point to get out of a pit. Moreover, in order to go downwards without too much roaming around, we have to reduce acceptance mutliplying variation of energy by `accept = 500`. This makes a probability of acceptance around 0.01 for a variation of energy around `0.05` (we roughly multiply minimum step by dimension to get a first order idea of energy variation since it's a polynomial function). This is a very high acceptance rate but it makes it more likely to fall in the right pit. The algorithm runs in about `170 sec` on my computer. Best energy value is between `100` and `220` with random data (see below for real data).

Afterwards, we implement gradient method on the best solution :

When working with random data, we fall in the pit nearby the optimal solution with a fitness around `100` and an error at `20` (relatively small, showing the difficulty to find the right point). Computation takes around `1 min`.

When working with real data, we perform simulated annealing `5 times` on the same function therefore the probability to get closer to the real optimal point is higher :  
Best energy values range between `77` and `1,041`. When we perform a gradient descent from the best point of the 5, we see that we still get stuck in a local optimum even if the performance is improved.

# Optimize shifted Rastrigin's function with simulated annealing algorithm, coupled with gradient local search

[Description of the function : CEC2008_TechnicalReport.pdf](CEC2008_TechnicalReport.pdf)  
[Data source : data.py](data.py)  
[OPEN MAIN CODE FOR DIMENSION 50 : Function_Rastrigin-50.py](Function_Rastrigin-50.py)   
[OPEN MAIN CODE FOR DIMENSION 500 : Function_Rastrigin-500.py](Function_Rastrigin-500.py)   
[OPEN UTILITIES : Functions_util2.py](Functions_util2.py)

For this section and the next one, utilities functions have been re-written based on previous experience to improve performance : extensive use of Numpy methods, handling of batch processing to take advantage of Numpy. And in the main codes, we have used parallel computing. Note that parallel computing is at the cost of a loss of reproducibility.

## Simulated annealing coupled with gradient descent in dimension `50`

First of all, we want to take advantage of numpy optimization (contiguous storage) working on arrays as much as possible. We work with batches of points to evaluate the Rastrigin function, to perform gradient descent and simulated annealing. Moreover, instead of calling the random generator each time we need a random vector, we compute a batch of random vectors in advance and we read it sequentially. And eventually, we parallelize computation.

Note that we have also modified simulated annealing acceptance probability function :  
it is now `exp(- accept * (variation of energy))`.  
It is closer to the original Kirkpatrick's algorithm but as if temperature was constant. We have worked on that side because the variations of Rastrigin function has a very broad range, often leading to extreme acceptance (all or nothing). We adjust the acceptance parameter according to the final variations : we expect the function to vary by a few units by the end of the process. If we choose `accept = 0.3`, we get a probability function around `exp(-0.3*2) = 0.5 approx` which is appropriate.

With these improvements, we can perform simulated annealing of length `10^6` from `100` initial points simultaneously, distributed among `all-1 = 11 cores` on my computer within around `6 min`. We set diversification factor starting at `1.5` and going down to `0.015` with exponential decrease.

We get following results before gradient descent :  
(note that it seems impossible to ensure reproducibility, probably due to parallelization)
```
  [[-210.0610746  -209.45773572 -230.52549112 -209.62921156 -190.51968507  
    -190.36260963 -188.63993378 -218.19881553 -197.69859899 -211.68365166]  
  [-235.56503707 -235.05730646 -202.71083269 -193.52546241 -221.84944724  
    -224.40611599 -214.8505377  -225.43431673 -192.62248439 -240.64062135]  
  [-221.04703102 -208.86431536 -224.09012305 -211.55884055 -188.18152277  
    -212.20287767 -245.88878574 -212.96466426 -227.42747413 -190.01316071]  
  [-192.45407258 -203.24994706 -175.9449979  -193.78551655 -208.86038389  
    -228.17744407 -236.041239   -194.75704598 -183.26436612 -196.59390383]  
  [-192.25150509 -225.66486805 -212.51786338 -197.40782674 -205.00467485  
    -215.37618134 -225.68183249 -186.38179699 -207.50768217 -202.29792478]  
  [-221.4125073  -194.49776703 -191.67898674 -232.48087651 -215.71460129  
    -184.0828504  -200.60814312 -216.26809388 -203.61491744 -240.83091924]  
  [-202.36565854 -227.26948268 -189.65893358 -230.42220686 -206.78822047  
    -203.39537269 -220.54214768 -213.30661978 -203.43388121 -239.02873733]  
  [-222.22844975 -221.19429241 -221.97592546 -225.62801673 -219.11010397  
    -192.39328556 -225.60249362 -233.41809101 -214.10318767 -209.52363567]  
  [-209.10306374 -218.40571801 -216.57319785 -212.10955402 -214.6702979  
    -207.49886476 -231.84223856 -214.99703031 -232.41673894 -239.43396927]  
  [-205.21311403 -228.8354484  -203.46301897 -229.06656103 -223.23423678  
    -223.74485579 -216.93262568 -222.5562782  -184.90988444 -214.81448915]]  
```

After gradient descent on each point (computation takes less than `1 sec` once parallelized), we get :

```
    [[-252.38814042 -251.39000607 -274.28088294 -255.37304303 -236.01570388  
    -229.49548267 -230.49914575 -250.38973004 -245.42842517 -253.38323011]  
  [-271.28446208 -275.27590988 -246.42122226 -235.46200128 -270.29597955  
    -260.34400593 -249.40486487 -263.32396439 -235.47172106 -279.24850875]  
  [-258.36003385 -257.35799856 -258.36081272 -246.41859247 -225.52924075  
    -249.40839948 -280.25115466 -250.40316048 -268.30755784 -235.47356573]  
  [-230.49111964 -244.42094789 -215.57526286 -232.48037585 -243.43517278  
    -265.88727333 -269.30339624 -231.04346807 -219.55685937 -234.47723107]  
  [-240.43736786 -261.33554147 -250.94207712 -235.4741206  -239.44916803  
    -251.39059052 -266.31780597 -225.52038989 -250.94190352 -241.44238867]  
  [-260.33538573 -231.49823063 -242.44182055 -273.28419284 -254.36639145  
    -234.47800861 -240.45352142 -259.35791772 -242.43006159 -281.24691273]  
  [-245.42385372 -261.34502945 -235.46173776 -270.3008456  -243.43607396  
    -246.42106465 -260.35084364 -255.36415239 -241.44390069 -280.24134451]  
  [-258.35677674 -258.36303038 -266.30469978 -268.30753229 -258.90152999  
    -230.4992622  -270.30049108 -269.30249739 -259.34364057 -248.40066231]  
  [-237.46385358 -252.3926271  -244.43371858 -255.36798072 -255.3781104  
    -243.42402871 -268.31030901 -257.35205459 -272.2907585  -283.2321993 ]  
  [-246.4154355  -272.288758   -239.45380715 -260.34879062 -264.32849538  
    -264.33197819 -244.43180547 -261.3310405  -225.52675914 -259.34670117]]  
  ``` 
Best fitness is now around `-280` to be compared with the targe `-330`. The cloud of points we get is much homogeneous.

We now try a gradient on each isobarycenter of each sample of 10 points. Again we get very homogeneous results :

```
  [-310.09008741 -308.10806078 -310.09773435 -308.10062952 -308.09683851  
  -308.09843743 -314.07787534 -307.1082976  -308.10802103 -310.10032396]
```
The minimum value is now around `-315` for a target at `-330`.

## Implement with real data in dimension `50`

Finally, we decide to adopt the following process :  
a) run simulated annealing on `50 samples of 10 initial points (10^6 iterations each) with 12 cores in parallel`.  
b) run gradient descent on each one of the `500` best points found.  
c) take the `50` isobarycenters of the `50` samples of obtained points.  

a) takes around `30 min`. Best fitness is around `-255`.  
b) takes `8 sec`. Best fitness is approximately `-295`.  
c) takes a few secs. best fitness is approximately `-315`.  
Final error is less than `4` which is quite reasonable.

This result is reasonable considering the machine we are using but it still looks difficult to extend it in dimension `500`.

## Extend the method to dimension `500`

Considering the huge effect of dimensionality on our previous method, we need to change the approach. Simulated annealing seems interesting but the fact that our `500` wandering points are completely independent considerably slows down the process. In order to introduce interdepence between the points so that they can help each other, we could use PSO algorithm but it gets stuck easily in local minima and we need more diversification. 

For these reasons, we would like to introduce a new process based on barycentric approach. We keep using the annealing idea but let us say that we cool down a gas of thousands of particles taking into account attraction/repulsion effects. At each step, for each point `P_0`, we perform the following operations :

- pick a random set of `neighbours` points different from `P_0` : `P_1, ..., P_neighbours`
- pick a set of `neighbours + 1 ` weights according to the following rule :
  * `w_0` weight of `P_0` is user defined, call it `min_main_weight` . It is set to ensure that we somehow stay in the neighbourhood of `P_0`.
  * all weights should sum to `1` but we want to allow negative weights for exploration outside the convex hull of the gas so we proceed in 2 steps according to a user defined parameter called `neg_margin`:
    + 1st step is iterative
    `w_1` is a random (uniform law) weight in the interval  
        `[ 0, 1 + 2 * neg_margin - w_0 ]`  
    `w_2` is a random (uniform law) weight in the interval  
        `[ 0, 1 + 2 * neg_margin - w_0 - w_1 ]`  
    ...  
    `w_neighbours` is a random (uniform law) weight in the interval  
        `[ 0, 1 + 2 * neg_margin - w_0 - w_1 - ... - w_(neighbours-1) ]`  
    + 2nd step is a phase of adjustment  
    Let `i` be the smallest index so that `w_0 + ... + w_i > 1 + neg_margin`  
    Set all weights of index greater than 1 to be negative.  
    Adjust `w_i` so that all weights sum to 1.  
- compute `P'_0` to be the barycentre of `P_0, ..., P_neighbours` with weights `w_0, ..., w_neighbours`  
- `P'_0` is accepted of rejected with the same acceptance rules as in simulated annealing  

### barycentric approach in dimension `500` with real data

- gas has `cloud_size = 250,000` particles as a tradeoff between diversity and computation capacity... It is not much considering the domain.
- `max_run = 1,000` this parameter is set so that particles can move in more directions (dimensionality issue)
- `accept = 0.1` this value is on the fitness function and does not change much with dimensionality
- `min_main_weight = 0.9`, `neighbours = 15` and `neg_margin = 0.8` : The number of neighbours is set so that the final adjustment step does not introduce any major perturabtion is the weights distribution. We have a large negative margin because it is difficult to explore out of the convex hull when the number of particles is big. 

To implement the procedure with real data, we parallelize the algorithm :  
We run `5 threads` in parallel, each one handles `50,000 points`. They run `50 iterations` in parallel after which the `250,000` points are reshuffled and again divided into `5 threads`. We proceed to `20 reshuffling`. As a result, the gas of `250,000 points` is moving by only `1,000 steps`. Indeed, on preliminary trials, we observed a very fast convergence of the cloud and going further leads to a big loss in diversity. Regarding parameters, we keep previous parameters that proved to be very efficient.

Running the algorithm takes around `4,000 sec` and we observe the best fit at each reshuffling :

```
  8224.000546831263  
  7779.056951473989  
  7491.566042214355  
  6292.424706570562  
  4345.641330468526  
  4064.237600988399  
  3959.0807908439783  
  3895.5360381051796  
  3800.3369723341875  
  3706.294926376571  
  3674.257215565347  
  3628.385904057478  
  3604.114899732278  
  3570.343456305375  
  3483.7688624973907  
  3411.2526402933145  
  3343.4136194135153  
  3202.746405318051  
  3074.5183153482617  
  2772.9842828384635  
```

We observe a first period of fast convergence followed by a plateau around `3,600` and again an accelerating convergence.

Unfortunately a gradient descent on the `250,000 best points` is too costly. So we take a random sample of `5,000` to perform the gradient descent. On high dimensions, the gradient values may have a very wide range and it creates a risk of failure. We use a small `ro = 0.005` (learning rate) and additionnaly clip any move in any direction in `[-0.2, 0.2]`. We run the algorithm in `5 parallel threads`. It takes `350 sec` to run. We see a good diversity of final fitness with a best one around `-230`.

We also try to take barycentres of 5 randomly chosen points in the gas. With `5,000 barycentres`, it takes more or less the same time to run (`350 sec`) for a similar best fitness. Both approaches appear to be receivable.  

Main problem is reproducibility : variability of the result is quite important.

## Comparison between dimension `50` and dimension `500`

In dimension `50`:  
We have used classic simulated annealing followed by gradient descent :  
simulated annealing takes `500,000,000 evaluations`  
gradient descent (including gradient approximation computation) takes :  
`550 points * 500 iterations * (1 + 50*2 for gradient) approx 28,000,000 evaluations` which is negligible compared with simulated annealing.  
The whole thing takes around `30 minutes` to run.

In dimension `500`:  
We have used a home made barycentric simulated annealing followed by gradient descent :  
simulated annealing takes `250,000,000 evaluations`  
gradient descents (including gradient approximation) takes :  
`10,000 points * 20 iterations * (1 + 500*2 for gradient) approx 200,000,000 evaluations` which is of the same order as the simulated annealing.  
The whole thing takes around `80 minutes` to run.  

Keeping in mind that the performance in dimension `500` is not as good as dimension `50` it seems that barycentric approach is quite efficient. In any case, we could not find proper parameters for classic annealing leading to comparable results in a reasonable time.

# Optimize shifted Griewank’s Function in dimension `50` and `500`

[Description of the function : CEC2008_TechnicalReport.pdf](CEC2008_TechnicalReport.pdf)  
[Data source : data.py](data.py)  
[OPEN MAIN CODE : Function_Griewank.py](Function_Griewank.py)   
[OPEN UTILITIES : Functions_util2.py](Functions_util2.py)

I have read somewhere that Griewank's function is easier to optimize in high dimension. Based on that assumption, I use a very simple version of a generalized gradient descent algorithm. In a first computation, I implement a descent on a cloud of points based on a gradient estimated with a very big step (`+/-5 on each coordinate`) and huge `ro` value, with almost no limitation on the size of the move. That way, I capture a global tendency of the function. Once this cloud is stabilized, I implement a classical approximated gradient descent on the barycentre of the cloud.

## Optimize shifted Griewank’s Function in dimension `50` with gradient descent

1st step :
- `cloud_size = 300`, it is a bit pessimistic but in dimension `50`, this precaution will not cost much.
- `eps = 5` is the size of the step to estimate the "pseudo-gradient" (in this case, we call it a gradient because of the method but, of course, it has nothing to do with a local mathematical gradient)
- `ro = 100` is the learning rate (`move = - ro * gradient`), it is set very high to move as fast as possible but still limitated by a `max_step = 30` (limitation is in euclidian norm value)
- initial cloud is chosen randomly

Computation takes `12 sec` and final value for the target function is around `-178` all accross the cloud, to be compared with the target value = `-180`.

2nd step :
- `eps = 0.01` : as small as possible but avoiding machine limitations effects
- `ro = 1` : we don't know the gradient's value but we keep a constraint of `max_step = 1` (euclidian norm on the move, respected by scaling)
- `max_run = 500` : as we have very small steps, we need to have a lot of steps to compensate (but it's ok since we are working with one only point)

Computation is very fast and final fitess is `-180`, the target value.  
Indeed, we can check that the final point is the real target point : euclidian norm of the difference is `10^-6`.

## Optimize shifted Griewank’s Function in dimension `500` with gradient descent

1st step :
- `cloud_size = 200`, it is a bit optimistic compared to previous case but in dimension `500`, computation is costly.
- `eps = 5` same reasoning as before
- `ro = 1000` limitated by a `max_step = 300`

Computation takes around `200 sec` and final values of the function are more heterogeneous, around `-175`.  
We are less optimistic than before due to dimensionality and we take the barycentre only on the best `100` points.

2nd step :
- `eps = 0.01` : as small as possible but avoiding machine limitations effects
- `ro = 1` : we don't know the gradient's value but we keep a constraint of `max_step = 1` (euclidian norm on the move, respected by scaling)
- `max_run = 500` : as we have very small steps, we need to have a lot of steps to compensate (but it's ok since we are working with one only point)

Computation takes `2 sec` and final fitess is `-179.99999948`, the target value being `-180`...  
Indeed, we can check that the final point is the real target point : euclidian norm of the difference is around `0.01`.