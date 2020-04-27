- [Metaheuristic-Optimization](#metaheuristic-optimization)
- [Travelling Salesman Problem (TSP) with `n = 38` cities in Djibouti (pyhton code see `TSP38.py`and its dependency `TSP_util.py`)](#travelling-salesman-problem--tsp--with--n---38--cities-in-djibouti--pyhton-code-see--tsp38py-and-its-dependency--tsp-utilpy--)
  * [Implementation of a genetic algorithm on TSP problem for Djibouti `n=38` (target is `6,656`)](#implementation-of-a-genetic-algorithm-on-tsp-problem-for-djibouti--n-38---target-is--6-656--)
    + [Improvement tries](#improvement-tries)
  * [Implementation of closest neighbour algorithm on TSP problem for Djibouti `n=38` (target is `6,656`)](#implementation-of-closest-neighbour-algorithm-on-tsp-problem-for-djibouti--n-38---target-is--6-656--)
- [Travelling Salesman Problem (TSP) with `n = 194` cities in Quatar (pyhton code see `TSP194.py`and its dependency `TSP_util.py`)](#travelling-salesman-problem--tsp--with--n---194--cities-in-quatar--pyhton-code-see--tsp194py-and-its-dependency--tsp-utilpy--)
  * [Implementation of a genetic algorithm on TSP problem for Quatar `n=194` (target is `9,352`)](#implementation-of-a-genetic-algorithm-on-tsp-problem-for-quatar--n-194---target-is--9-352--)
  * [Implementation of closest neighbour algorithm on TSP problem for Quatar](#implementation-of-closest-neighbour-algorithm-on-tsp-problem-for-quatar)
- [Conclusion regarding TSP problem with genetic algorithm and best neighbour](#conclusion-regarding-tsp-problem-with-genetic-algorithm-and-best-neighbour)
- [Optimize shifted sphere function in dimension `d = 50` and `d = 500`](#optimize-shifted-sphere-function-in-dimension--d---50--and--d---500-)
  * [Gradient descent algorithm for unknown shift](#gradient-descent-algorithm-for-unknown-shift)
  * [Particle sworm optimization](#particle-sworm-optimization)
  * [Conclusion regarding shifted sphere with gradient descent and particle sworm.](#conclusion-regarding-shifted-sphere-with-gradient-descent-and-particle-sworm)
- [Optimize shifted Schwefel's problem 2.21 in dimension `d = 50` and `d = 500`](#optimize-shifted-schwefel-s-problem-221-in-dimension--d---50--and--d---500-)
  * [Gradient descent algorithm for unknown shift](#gradient-descent-algorithm-for-unknown-shift-1)
  * [Particle sworm optimization](#particle-sworm-optimization-1)
  * [Conclusion regarding shifted Schefel's problem with gradient descent and particle sworm.](#conclusion-regarding-shifted-schefel-s-problem-with-gradient-descent-and-particle-sworm)
- [Optimize shifted Rosenbrock's function with simulated annealing coupled with gradient method](#optimize-shifted-rosenbrock-s-function-with-simulated-annealing-coupled-with-gradient-method)
- [Optimize shifted Rastrigin's function with simulated annealing algorithm, coupled with gradient local search](#optimize-shifted-rastrigin-s-function-with-simulated-annealing-algorithm--coupled-with-gradient-local-search)
  * [First tests on random data in dimension `50`](#first-tests-on-random-data-in-dimension--50-)
  * [Improve algorithm efficiency (still on random data) in dimension `50`](#improve-algorithm-efficiency--still-on-random-data--in-dimension--50-)
  * [Implement with real data in dimension `50`](#implement-with-real-data-in-dimension--50-)
  * [Extend the method to dimension `500`](#extend-the-method-to-dimension--500-)
    + [test barycentric approach in dimension `50`](#test-barycentric-approach-in-dimension--50-)
    + [test barycentric approach in dimension `500`](#test-barycentric-approach-in-dimension--500-)
    + [implementation with real data](#implementation-with-real-data)
  * [Comparison between dimension `50` and dimension `500`](#comparison-between-dimension--50--and-dimension--500-)
- [Optimize shifted Griewank’s Function in dimension `50` and `500`](#optimize-shifted-griewank-s-function-in-dimension--50--and--500-)
  * [Optimize shifted Griewank’s Function in dimension `50` with gradient descent](#optimize-shifted-griewank-s-function-in-dimension--50--with-gradient-descent)
  * [Optimize shifted Griewank’s Function in dimension `500` with gradient descent](#optimize-shifted-griewank-s-function-in-dimension--500--with-gradient-descent)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Metaheuristic-Optimization
8 problems to optimize
Using Python 3.6 with Numpy 1.17, Scipy 1.4.1, psopy 0.2.3.

# Travelling Salesman Problem (TSP) with `n = 38` cities in Djibouti (pyhton code see `TSP38.py`and its dependency `TSP_util.py`)

Data source : [National Traveling Salesman Problem, University of Waterloo (Canada)](http://www.math.uwaterloo.ca/tsp/world/dj38.tsp)

We are looking for the shortest path so as to visit all cities without visiting twice the same.  
We know the coordinates of each city on a 2D euclidian plane.  
We first build a distance matrix, using usual 2-norm. This is a symmetric matrix with null diagonal and strictly positive values everywhere else satisfying the triangular inequality. The computation is asymptotically of order `n^2`.

A solution of our problem is a specific arrangement of `[1,n]` . It can be viewed as a hollow `nxn` matrix where `x_(i,j) = 1` whenever the path includes going from city `i` to city `j` . It has only `n` non-zero values (return to origin) and it is of rank `n`. The total distance function to minimize is the sum of the product of the distance matrix by the solution matrix. Obviously, this is a very simple product and it can be simplified summing elements read one by one in the distance matrix. It goes from an order `n^3` to an order `n` algorithm. Together with the generation of random solution, it is the main purpose of the [`TSP`class](https://github.com/jMetal/jMetalPy/blob/master/jmetal/problem/singleobjective/tsp.py) proposed in `jmetalpy`. We use it. Note that it also suggests a method to compute the distance matrix but it is far from being optimized compared to `scipy` method which uses a general optimized with numpy arrays [Minkowski distance](https://github.com/scipy/scipy/blob/v1.4.1/scipy/spatial/kdtree.py#L15-L55). We therefore create a new class to adjust to our local problem. It will be of no consequence for using  `jmetalpy` later on because we keep all attributes and public methods.

It is a discrete non-linear problem and we can say that one solution is related to another one by a series of transpositions (any permuation can be decomposed in transpositions). This kind of modification is strongly relevant in a genetic context. It seems relevant to try and use a genetic algorithm in this context. We must be careful with this method because it may not converge and indeed, [as shown in 2005 by H. Abdulkarim and I.F. Alshammari](https://www.researchgate.net/publication/280597707_Comparison_of_Algorithms_for_Solving_Traveling_Salesman_Problem), it happens that it does not converge in the TSP case. We could also tried Simulated Annealing, Tabu Search, Particle Sworm Optimization that fits we the discrete non-linear problems. But we have found in litterature that in TSP problem, there are many local optima and genetic algorithm is known to be indifferent to these issues. For instance, [in 2012 W. Hui showed](https://www.sciencedirect.com/science/article/pii/S2211381911002232) (note that these are a conference proceedings and english syntax is very poor) that for `n=51` we can get an excellent result with an ant colony algorithm but it is very sensitive to parameters. As I am not well versed in this subject, I'd rather use a more robust method. 

## Implementation of a genetic algorithm on TSP problem for Djibouti `n=38` (target is `6,656`)

Let `D` be the distance matrix.

The number of possible paths of the travelling salesman is of order `(n-1)!` which in our case amounts to `10^34` which is huge. On the other hand, the length of the pathe is bounded between `n * min(D+)` and `n * max(D+)` where `D+` is the set of strictly positive values of `D`. In our case, the bounds are roughly `200` and `70,000` . It is necessary to choose a population size that is significant with respect to the size of our problem but keeping in mind that the complexity of the algorithm is directly proportional to the population size. We choose `n_0 = 1,000` with a maximum number of iterations at `20,000` this large population is likely to favor diversity of our population. We keep in mind that in Hui computation, the optimal result has been found after less than `400` iterations for `n = 51` even if the effective stopping criterion has been the number of iterations (`2,000`). We could refine our model choosing a stopping criteria as a number of iterations without improvement.

Regarding mutation procedure, the minimum change that can be introduced in a solution is a transposition. We can either choose the permuation randomly or optimize this process (which introduces an overhead). We choose it randomly to begin with, that will save computation. We take Hui's probability at `0.01` .

Regarding crossover, we keep the classical method in litterature. We swap 2 sections among parents (swap genes of random size) and replace remaining alleles in other slots according to original sequential order :
P1 = (1,  2,3,4,5,  6,7) ; P2 = (4,  3,1,2,7,  6,5)
swap from index 2 to 5 :
S1 = (1,  3,1,2,7,  6,7) ; S2 = (4,  2,3,4,5,  7,5)
re-arrange :
S1 = (4,  3,1,2,7,  5,6) ; S2 = (1,  2,3,4,5,  7,6)
We also take Hui's probability at `0.3` .

We use a roulette wheel selection that a probability to mutate or mate inversely proportional to the length of the trajectory. Probability is proportionnal to the trajectory length. Note that the `RouletteWheelSelection` class proposed in `jmetalpy` has to be reshaped as it is suited only for maximization problem.

We set random seed at 1203 for reproducibility.

Based on these results, we get a first result at `15,976`.

### Improvement tries

We first increase the selection severity taking a probability depending exponentially on the trajectory length. The result is not improved at `17,983` . It shows that intensification does not improve the result. We should rather try to diversity.  
Now, instead of choosing a random mutation, we choose the mutation of best improvement. It is a bit costly in terms of computing (order `n` with optimized algorithm) but it will give a path that may be further from the parent. Score is `12,760`, this is a real improvement, we keep the optimization.  
If we cumulate both changes, we get a fitness at `14,224` . There is no improvement.

We change our strategy. We know from Dr Nakib that genetic algorithm might perform well with small population. Therefore we leave Hui's strategy to have a large population and reduce our population down to `100` . Number of evaluations is proportionally increased up to `100,000` . Fitness is `10,694` . This is a significant improvement.  
We try to work on this approach. Considering the population decrease, we have a mutation probability that give a number of mutation at each generation close to `0` . This is not good for diversity. We change its order of magnitude up to `0.5` . New fitness is `8,091` which is again a significant improvement. Computation takes `22 sec` on my computer.  

At this point, we re-run our programm with another seed (`1,815`)to check that improvements are not specific to our case. Things are OK, fitnesses follow the same evolution. We come back to our initial seed.

Considering the high variability of genetic algorithm, we perform the computation `100` times and take the minimum result, that is `7,025`. We stop our quest at this point. Considering that the target is `6,656` and our simple PC computation capacity, it seems difficult to go much further.

## Implementation of closest neighbour algorithm on TSP problem for Djibouti `n=38` (target is `6,656`)

This is a deterministic algorithm known to be quite efficient as a first approach. We initiate the algorithm from each node and find the best path from this node. The best of the `38` paths has length `6,770`. However, since `n` is small, we can optimize thoses path by best transpositions. In that case, the best path has length `6,659` which is a very good performance. We keep it.

Eventually, this last method is more efficient than genetic algorithm but without any chance to reach the best path. Genetic algorithm is less efficient on average but gives us a chance to reach the best path.

# Travelling Salesman Problem (TSP) with `n = 194` cities in Quatar (pyhton code see `TSP194.py`and its dependency `TSP_util.py`)

## Implementation of a genetic algorithm on TSP problem for Quatar `n=194` (target is `9,352`)

The major issue of this new problem is the increase of `n` and consequently the high computing demand. In this case, the target is `9,352` with rough bounds for our trajectories `193` and `281,600` .

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

Note that last computations takes 1 min to 2 min on my computer. I did implement the optimal mutation but the computation demand is high. I also noted that we get similar results with population of 30 and offspring of 10 individuals.

We roughly have the same performance profile except that last result is quite far from optimal solution and we note that exponential method for roulette wheel selection might be efficient for large populations but not for small populations.  
This genetic algorithm is costly and no so efficient as far as we can see. 

## Implementation of closest neighbour algorithm on TSP problem for Quatar

 We try the closest neighbour path and we find a trajectory length equal to `11,330` which is quite good considering the target `9,352` and our previous result.

We can optimize this path with best transpositions to get it down to `10,877` by deterministic best mutations or when we randomize the choice of mutation among the best ones. It means that this point is quite attractive. Or `10,837` if we completely randomize the mutation.  
I did not manage to get a significantly improved result.

# Conclusion regarding TSP problem with genetic algorithm and best neighbour

We see that a genetic algorithm does not require a large population to perform. This seems in contradiction with a lot of litterature and it is counter-intuitive. We also observed that it is difficult to predict the effect of selection process. However, it is clear that the use of an optimized mutation is very useful in both cases even if the computational cost can not be neglected. It could be interesting to study other methods like different types of crossover or different types of mutations or introducing new individuals every so many generation.

As a remark, we can also note that the package `jmetalpy` is in many aspects not optimized and may even contain small mistakes. Hovever, the code is simple enough to be proofread.

We also observed that the best neighbour algorithm is very efficient, especially if combined with a best transposition descent algorithm. Finding the best path from a given city is a computation of order `n` . Finding the best path among all cities is of order `n` therefore the algorithm is of order `n^2` . Finding the best mutation for a given path should be of order `n^2` (test each mutation among `n(n-1)` ) but with numpy optimized arrays, it is possible to reduce it to get a power of `n` between `1` and `2` . I don't know about the length of the descent but it should somehow grow with `n` too.

# Optimize shifted sphere function in dimension `d = 50` and `d = 500`

For reproducibility, results are presented with `random.seed(2610)` and `rng = default_rng(1611)` .

The bias is set to `-450`, the shift may vary within the hypercube centered on the origin with side length `200` .  
We look at the function as a black box.  
This is a non-linear real continuous function. We can first use a gradient method with estimated gradient 

## Gradient descent algorithm for unknown shift

We test our method with random values of the shifts to be sure that our result does not depend on the specificity of the given data. Indeed, this method is deterministic and it does not make sense to test it `25` times on the same initial value.

Usually, the gradient method is used when we know the gradient function but it is possible to implement it with an estimated value of the gradient. The cost is of order `d` . We get an excellent result in `2` or `3` runs in dimension `50` and `3` runs in dimension `500`. Note that this algorithms can also be used for a local optimization after a global search with another method. For a real function, it seems important to me to try this method before anything.

As a global optimization, this gradient can be accepted only if we know at least that the function is unimodal.

## Particle sworm optimization

We know that the algorithm is very sensitive to parameters. We therefore make a quick trial with different learning rates.  
We observe that for this unimodal function, the highest is the global learning rate, the best is the result, independently of local learning rate. Conversely, in case of unknown function, this may be an indication that our function is most likely unimodal.

Based on this trial, in dimension `50`, we take `g_rate = 5` and `l_rate = 0.2` . PSO, unlike genetic algorithm, does not introduce much diversity, therefore we need a large initial population, set at `500` . For computational limitation reasons, we set `max_iter = 1,000` . Considering our window, the hypercube, we keep maximum velocity at `5 units/iteration` . We leave friction at default value `0.5` because we have observed fast convergence when testing learning rates.

Each run takes around `10 sec` on my computer. Results are quite close to optimality, we are mostly within a few units of the minimum which is `-450` . Which is good considering the huge value range of the objective function.

In high dimension, checking constraints is very costly, that's why it is cheaper and as efficient to introduce a penalizing term. We introduce a penality that is not huge (not like a "wall") to avoid any unwanted side effect considering that somme coordinates of the shift (x target) might be close to the border.

Considering the high dimensionality, we will have to boost the positive effects of global learning rate because of the noise introduced by dimensionality. We make a test similar with the previous one but with greater range. We observe the same global behaviour. Because of dimensionality, we also decide to increase number of iterations to `1,500` even if the cost is heavy considering a population of `1,000` . We take a global learning rate equal to `7`, keep a local learning rate at `0.2` . But because of dimensionality, we reduce maximum velocity down to `1` because the maximum velocity is per coordinate, therefore its effect is multiplied by dimensionality. In order to balance out this limitation, we slightly increase friction so that the sworm does not stop before the optimum, say `0.9` . Knowing that those parameters may quickly move an individual out of the domain, we take initial values in a slighty reduced hypercube to avoid useless search outside the domain.

Because of computational cost, we run the algorithm only `5 times` . 

The algorithm converges each time but `1,500` iterations is not enough to reach the target. However, as mentioned before, we can make a local search from this result with gradient descent algorithm. We get successively `-135, -153, -160, -170, -167` . But average time to reach those results is `700 sec` on my computer, that is 70 times more than is dimension `50` .

## Conclusion regarding shifted sphere with gradient descent and particle sworm.

We observe that gradient descent is thousand times more efficient than particle sworm in this case. But we must remember that gradient usually converges only locally to the nearest local optimum and that it may not converge at all. On the other hand, setting parameters for PSO algorithm is quite touchy and algorithm is heavy but it usually ensures a global search.

# Optimize shifted Schwefel's problem 2.21 in dimension `d = 50` and `d = 500`

Conditions and reproducibility seeds are the same as before.

## Gradient descent algorithm for unknown shift

We must be careful because this is not a smooth function. Therefore we compute a pseudo-gradient. We have a function that is continuous but only piecewise derivable. It can be interesting in this case to "jump" over those discontinuites : we would compute the pseudo gradient `f(x + eps/2) - f(x - eps/2)` with `eps` greater than usual. Instead of taking `eps << 1`, we take a value arount `1` . Obviously we loose a lot of precision in this process and it can be of interest to rescale `eps` along the process.

We can anticipate that the gradient algorithm will be less efficient than previously because it will move one coordinate after the other. In that case, it can be interesting to have a coefficient `ro` (the coefficient multiplying gradient in the descent step) that will be adjusted along the process : start with big leaps to move out of those "plains" and get more precise when approaching the target. That is why I decide to start with `ro = 7` during half of the iterations, `ro = 0.7` during a quarter of the iterations, `ro = 0.07` during a quarter of the iterations. Initial `ro` value is chosen as a parameter.  
I have also tried an algorithm with optimized descent, adjusting the leap length at each step but is gives no satisfactory result.

In dimension `50`, we keep a fixed small value for `eps`, with `2,000` iterations, we get a decent convergence with end values between `-449,96` and `-450` when the target is `-450` . The computation cost is low, taking aroung `1 sec` on my machine. Since we know that the algorithm converges we can run it longer with smaller values of `ro` to get more precision at low cost.

In dimension `500`, we take an initial value of `eps = 1` and rescale it twice is the same proportions as `ro` . With `3,000` iterations starting from `ro = 50` we get end values between `-449.6` and `-449.9` when the target is `-450`. Each run takes aroung `24 sec` on my machine. Because of dimension, function evaluation time is multiplied by `10` . As we know the function, we know that search will also be complexified.

I can also use simulated annealing, tabu search or particle sworm optimization among others. In order to compare with the previous case, I will keep PSO.

## Particle sworm optimization

We we test global and local learning rates, we observe a very different pattern from the previous function. We have an optimal zone along a line joining `(local rate = 2.5, global rate = 5.5)` with `(local rate = 5.5, global rate = 2.5)` , we choose `4` for both as a compromise.

We can observe that in dimension `50`, the algorithm has a poor convergence. Even if we easily reach a fitness equal to `-400`, the end of the job is very costly. This is a typical case were it would be interesting to couple this algorithm with a local search with a gradient descent. PSO algorithm tells us that we are probably close to a global optimum. Then we switch to a deterministic algorithm to find the nearest local optimum, hoping that it will be global.

We take a population of `50`, limit our algorithm to `1,500` iterations and we get an average value with 25 trials around `-405` with a convergence in all cases. We can take the average values of limits as an initial vector for a gradient algorithm. We get a performance similar to the pure gradient algorithm (`-449,96`) but with a convergence after `1,000` iterations only. It is therefore interesting to couple global and local search algorithms.

In dimension `500` it is quite difficult to identify a clear optimal zone for local and global learning rates. In fact, after a few tries, we see that it is very difficult to get even a decent convergence. The algorithm gets stuck around a  fitness equal to `-360` due to loss of diversity. I also observe that in spite of introduction a penalizing term to stay in the bounds, the "sworm" has a tendency to go out of control. If I try to apply the constraints directly, the computation overhead is too much due to poor optimization of the algorithm. Therefore I introduce a `callback` function to clip the vector in the domain. Convergence is much improved as far as the variable is concerned.

 In order to keep diversity, I prefer running quickly (`2 to 3 sec` each) the algorithm `100` times to generate a new initial population to be used for a new run. I keep previous parameters for the first part. Regarding the second part, I reduce `max_velocity` down to `0.5` and take default value `0.5` for `friction`, as we should already be close to the solution. But unfortunately, it is not conclusive.

In dimension `500` it is therefore much more interesting to use only gradient descent algorithm as PSO is way to heavy to handle. I get no interesting results below a population of `10,000` .

## Conclusion regarding shifted Schefel's problem with gradient descent and particle sworm.

As a conclusion we observe that it is interesting to couple gradient descent with PSO in dimension `50` but only the gradient descent gives good performance in dimension `500` . It would be necessary to run it from a large number of initial points to have a high probability to get a global optimum.

# Optimize shifted Rosenbrock's function with simulated annealing coupled with gradient method

Rosenbrock function is multimodal, therefore it is useless to try gradient method before a global search.  
Moreover, when we compute the gradient of the function, we observe that in addition to the global optimum at `x = o`, there is an area where gradient is null for all coordinates except the first one. At this point, all coordinates of `x-o` are equal except the first one. This means that it is a very deep pit, all the more attractive that dimension is high. Indeed, we'll see that Rosenbrock function is easily optimized in dimension `50` but in dimension `500`, we keep falling in the nearby pit.

We are trying to minimize it with a personnalized version of simulated annealing. As there is too much dependency between parameters in usual implementations, I find it easier to work with less parameters : we decide a maximum number of iterations (`maxIter`), the initial step (`divers`), the final step  (`intens * divers`) and a parameter for acceptance (`accept`).  
Perturbation is a simple random vector in `[-1,1]`. We scale it with factor `divers * intens^(iter/maxIter)`.  
Probability of acceptance is `exp(- nrjVar * accept)` where `nrjVar` is a positive variation of energy.  
This is quite simple but turns out to be quite efficient. That way we don't need to define a temperature which is often redundant with iterations count.

In dimension `50` we set `maxIter = 500,000` to limit computation time as much as possible.  
We set initial step at `divers = 0.3` and final step at `divers * intens = 0.000,3`, this is a reasonable tradeoff between the necessety to move in a range of `200` and to localize solution at a precision less than `0.1` . We keep `accept = 1` (Metropolis–Hastings setting). Each run takes around `20 sec`. We run the algorithm 5 times on random data. Best values are between `-330` and `-360` which is good to take to initiate a gradient. We keep those parameters to work on real data, we get same kind of results. But it is worth noticing that it is useful to run the algorithm several times as the variability is not negligible. We perform an optimized gradient method on the best result and we get a much better result within less than `10 sec` . We approach real solution by less than `0.2` on each coordinate and get a final Rosenbrock function value inferior to `-389.98` for a target at `-390`.

In dimension `500` we have to significantly lengthen the flight at `maxIter = 3,000,000`, we keep initial step at `divers = 0.3` but due to dimensionality effects, we scale down the final step at `intens * divers = 0.000,03` . Obviously, this might become an issue at some point to get out of a pit. Moreover, in order to go downwards without too much roaming around, we have to reduce acceptance mutliplying variation of energy by `accept = 500`. This makes a probability of acceptance around 0.01 for a variation of energy around `0.05` (we roughly multiply minimum step by dimension to get a first order idea of energy variation since it's a polynomial function). This is a very high acceptance rate but it makes it more likely to fall in the right pit. The algorithm runs in about `170 sec` on my computer. Best energy value is between `100` and `200` .
Afterwards, we implement gradient method on the best solution and we fall in the pit nearby the optimal solution.

Possible improvements : I could not go further by lack of time but it could be interesting to try to build a sworm with vectors found through simulated annealing to try to launch a PSO method to look around and try to find the best optimum location.

# Optimize shifted Rastrigin's function with simulated annealing algorithm, coupled with gradient local search

## First tests on random data in dimension `50`

In dimension `50`, we start with random shifts to test different methods. We perform `100` runs of simulated annealing limited to `3.10^6` iterations, with initial step set at `1` and final step set at `5.10^-3` (we start with a global search and finish with a local search). We set `accept = 1` which means, according the function range, that we refuse almost any increase at the beginning and start tolerating increase much later, when step is way smaller. We get a cloud of points that we expect to be around the global minimum. Each run takes `2 min` on my computer. And we get minimum energy list :  
```
[-120.16358571496738, -75.78344978772745, -69.05819743302106, -120.51101265994541, 15.850243165490951, -66.2992038050138, -118.76359419663757, -196.9227649160092, -131.98193670417604, -165.69228836145305, -80.01258139949391, -45.684349457413475, -113.26725434161136, -156.24593316589466, -138.37596488237168, -41.49974079854729, -50.842659537165446, 7.122330650211495, 1.1982858400099303, -94.64811150435241, -107.60280654777571, -15.652510824884416, -166.83432190723494, -101.86074842343703, -166.92730272418243, 54.95834131861625, -28.43493516330659, -105.26553842527272, -96.38425801314699, -81.73791181375472, 22.26141285574232, -146.21819635682647, -100.01965261103479, 22.856666260565703, -64.56363810803634, -83.4438617672198, -149.91873449004484, -100.85564763041052, -181.40944279330145, -84.4457826693125, -74.09627923528424, -85.22071155078743, 149.07260173404165, -110.05082606144359, -179.80777710804023, -33.798410152110876, -137.3686507818592, -74.42454348687983, -141.30564747871747, -173.04640296607565, -25.289247632869035, -129.9339705162757, -152.95586834673563, -79.80685419052338, -20.453974745029598, -98.0886910321504, -147.5991437466542, -105.51708167147859, -111.81866710129668, 36.63819941093897, -55.43472810316973, 67.61374900652487, -100.61266339572802, -133.81846229318103, -66.75620385748891, -49.63335141600646, 59.5390907990888, -127.46016953424524, -115.85399817147305, -163.61966908437023, -175.7117622454923, -142.53627718664282, -44.98603143687046, -81.39388816472459, -148.84293506098905, 49.27503594444988, -44.354321588862604, -87.50630063035277, -109.39516178315787, -47.13262868381497, -112.19242423930436, 90.72192121182047, -103.78234078929322, -108.54196427394425, 124.234008601776, 95.2554398286054, -182.88180737802725, -112.45037145686592, -133.7115663824237, -152.3316942271468, -77.95316528692845, -104.54069275913022, -84.5147276219354, -33.4770048346881, 73.33170536218938, -108.88016222049274, -195.92109914378713, -206.1068712304754, -104.38088817832252, -65.47622972697161]
```

We run the gradient algorithm on each one of the corresponding best found state. The best result yields a fitness of `-213` to be compared with the target at `-330`. Each run take `2 sec`.  
We try to improve this result performing the gradient descent on the barycentre of the 100 points previously found. And indeed, we find a fitness at `-264` which is much better.  
Now we try to make a local search in the area of the cloud found thanks to simulated annealing + gradient method. We take random points computed as a weighted sum of the points in the cloud. We limit the weights to be mainly positive to stay in the area of the cloud. We take random weights ranging from `-0.7` to `1`. We choose `100` random points `400` times and we perform gradient method on each one of these `40,000` points. Fitness comes to `-289` within `1,500` seconds.  
Now, we use directly the points from simulated annealing without gradient descent and use the same system of random weighted points, an identical number of times (`40,000`), fitness comes to `-290`, meaning that gradient optimization on the cloud given by simulated annealing is useless.

As a preliminary conclusion, this method sounds interesting but our algorithm is not optimized at all and considering our machine abilities, we have to improve efficiency if we want to get a chance to optimize the function in dimension `500`.

## Improve algorithm efficiency (still on random data) in dimension `50`

First of all, we want to take advantage of numpy optimization (contiguous storage) working on arrays as much as possible. We work with batches of points to evaluate the Rastrigin function, to perform gradient descent and simulated annealing. Moreover, instead of calling the random generator each time we need a random vector, we compute a batch of random vectors in advance and we read it sequentially. Based on previous runs, we also try to improve simulated annealing parameters : we give more space for diversity, ranging steps from `1.5` to `10^-2`. And eventually, we parallelize computation.

Note that we have also modified simulated annealing acceptance probability function : it is now `exp(- accept * (variation of energy))`. It is closer to the original Kirkpatrick's algorithm but as if temperature was constant. We have worked on that side because the variations of Rastrigin function has a very broad range, often leading to extreme acceptance (all or nothing). We adjust the acceptance parameter according to the final variations : we expect the function to vary by a few units by the end of the process. If we choose `accept = 0.3`, we get a probability function around `exp(-0.3*2) = 0.5 approx` which is appropriate.

With these improvements, we can perform simulated annealing of length `10^6` from `100` initial points simultaneously, distributed among `all-1 = 11 cores` on my computer within around `350 sec (5 to 6 min)` to be compared with `200 min` previously. And as a bonus, we get much better results before gradient descent : 
(note that it seems impossible to ensure reproducibility, probably due to parallelization)
```
[[-234.68756923 -227.64604969 -229.89489208 -198.64195857 -207.00626779
  -221.68101912 -197.18636792 -202.67040799 -229.34420727 -214.28598939]
 [-208.8472549  -215.98526897 -208.53485224 -199.30800331 -216.13100038
  -211.50693429 -217.08257198 -245.37630407 -254.43046222 -227.32373786]
 [-214.33852953 -209.65050322 -206.18389261 -218.2307218  -224.44798904
  -222.70232706 -234.28046813 -204.50327039 -181.3382825  -207.75945968]
 [-219.64847949 -179.337407   -203.94741943 -226.829812   -229.20923679
  -201.81425655 -221.72336209 -214.87371576 -224.02556841 -222.89603934]
 [-201.46615175 -221.99042672 -202.35988258 -212.59395544 -211.79263127
  -235.76358081 -206.08363067 -219.2647507  -193.20205265 -210.29579017]
 [-237.72700204 -217.28835296 -224.45998297 -241.00637057 -228.5609258
  -197.33895052 -215.99739401 -219.77214566 -239.00494049 -228.50175284]
 [-215.51338781 -226.23877563 -219.61748269 -181.43560892 -217.27726226
  -214.17499908 -209.14571062 -234.08491563 -227.83386023 -176.4791964 ]
 [-213.83139704 -221.8446324  -232.0801542  -201.17864226 -208.46829974
  -229.76099714 -225.84126761 -206.9675312  -200.56586526 -204.21920663]
 [-208.87937635 -212.41861397 -210.04535944 -209.55789026 -242.99497191
  -222.52313874 -215.78539701 -187.0668032  -206.19642008 -222.42306892]
 [-208.67915872 -206.21952221 -220.73134742 -232.98655841 -209.52897475
  -211.36964271 -210.13771285 -235.16943446 -182.71139556 -218.24992282]]
```

After gradient descent on each point (computation takes less than `1 sec` once parallelized), we get :
```
[[-277.26233205 -265.32283492 -266.31979953 -232.49398518 -242.43868691
  -255.36546862 -246.41005167 -243.43273271 -269.30401957 -255.37647478]
 [-240.43825119 -250.40215157 -252.38855332 -240.44890176 -253.38483461
  -256.35442112 -253.38825606 -280.23719313 -283.22564054 -267.31618709]
 [-249.39907914 -252.38508031 -240.45218005 -260.33951598 -271.29554023
  -258.3444906  -274.2775367  -238.46385874 -225.52343776 -248.39999773]
 [-266.31948181 -220.54666378 -239.45598341 -263.33271752 -263.33666722
  -239.45760486 -264.32717782 -254.37671229 -257.35575672 -272.28629038]
 [-239.99750611 -255.37583833 -238.45464185 -253.38316509 -252.39329074
  -272.28762161 -246.42351887 -260.35033308 -223.53459324 -247.41848513]
 [-272.28781236 -252.3896226  -262.33681854 -279.2520231  -266.31836487
  -240.44723534 -256.37184701 -255.36640872 -274.26374461 -270.28651855]
 [-252.39297592 -261.34293237 -266.31958989 -219.55251271 -261.34688476
  -243.42539968 -253.38648915 -276.26403173 -271.29488971 -220.55260689]
 [-251.39638704 -261.3405658  -270.29743837 -237.46642624 -252.39306644
  -269.3072664  -254.37735868 -248.4062254  -237.46812556 -250.39822038]
 [-243.43367733 -253.38344701 -254.37984894 -246.42319359 -284.22221142
  -261.88658166 -254.37053459 -228.49700139 -250.39867771 -269.30236649]
 [-246.42115688 -246.4232485  -258.36055456 -272.2849173  -254.37623673
  -248.41101043 -244.43268551 -278.25602937 -223.53922452 -256.36810194]]
  ``` 
Best fitness is now around `-280`, comparable with `-264` previously. But the other good point is that the cloud of points we get is much more homogeneous than before, probably due to better parametrization of simulated annealing.

We now try a gradient on each isobarycenter of each sample of 10 points. Again we get very homogeneous results :
```
[-311.08117533, -315.0748219 , -309.10245315, -317.0562309 ,
       -314.0797713 , -314.07849263, -310.09575191, -313.07342075,
       -311.07938521, -314.07929953]
```
The minimum value is now around `-315` for a target at `-330` to be compared with `-290` previously and that is achieved with much less computation.

## Implement with real data in dimension `50`

Finally, we decide to adopt the following process :
a) run simulated annealing on `50 samples of 10 initial points (10^6 iterations each) with 12 cores in parallel`.
b) run gradient descent on each one of the `500` best points found.
c) take the `50` isobarycenters of the `50` samples of obtained points.

a) takes around `30 min`. Best fitness is around `-260`.
b) takes `20 sec`. Best fitness is approximately `-290`.
c) takes a few secs. best fitness is approximately `-315`.

This result is reasonable considering the machine we are using but it still looks difficult to extend it in dimention `500`.

## Extend the method to dimension `500`

Considering the huge effect of dimensionality on our previous method, we need to change the approach. Simulated annealing seems interesting but the fact that our `500` wandering points are completely independent considerably slows down the process. In order to introduce interdepence between the points so that they can help each other, we could use PSO algorithm but it gets stuck easily in local minima and we need more diversification. 

For these reasons, we would like to introduce a new process based on bayrcentric approach. We keep using the annealing idea but let us say that we cool down a gas of thousands of particles taking into account attraction/repulsion effects. At each step, for each point `P_0`, we perform the following operations :
- pick a random set of `neighbours` points different from `P_0` : `P_1, ..., P_neighbours`
- pick a set of `neighbours + 1 ` weights according to the following rule :
  * `w_0` weight of `P_0` is user defined, call it `min_main_weight` . It is set to ensure that we somehow stay in the neighbourhood of `P_0` (in our case, we choose `0.9`).
  * all weights should sum to `1` but we want to allow negative weights so we proceed in 2 steps according to a user defined parameter calle `neg_margin`:
    + 1st step is iterative
    `w_1` is a random (uniform law) weight in the interval `[ 0, 1 + 2 * neg_margin - w_0 ]`
    `w_2` is a random (uniform law) weight in the interval `[ 0, 1 + 2 * neg_margin - w_0 - w_1 ]`
    `w_neighbours` is a random (uniform law) weight in the interval `[ 0, 1 + 2 * neg_margin - w_0 - w_1 - ... - w_(neighbours-1) ]`
    + 2nd step is a phase of adjustment
    Let `i` be the smallest index so that `w_0 + ... + w_1 > 1 + neg_margin`
    Set all weights of index greater than 1 to be negative.
    Adjust `w_i` so that all weights sum to 1.
- compute `P'_0` to be the barycentre of `P_0, ..., P_neighbours` with weights `w_0, ..., w_neighbours`
- `P'_0` is accepted of rejected with the same acceptance rules as in simulated annealing

### test barycentric approach in dimension `50`
We test our new method on random data in dimension `50` : 
- gas has `cloud_size = 5,000` particles to ensure diversity
- `max_run = 2,000` : since our process is computationally demanding, we want it to converge very fast or it is garbage
- `accept=0.1` : considering the reduced number of steps, we don't think it will be efficient to scale acceptance according to iterations count, moreover we expect the cloud to shrink and by barycentration, steps will shrink too. We choose a "middle way" between `1.5` and `10**-2` we had before. 
- `min_main_weight = 0.9`, `neighbours=10` and `neg_margin=0.5` : these parameters are chosen so that the neighbours'weights are way smaller than `min_main_weight` (to keep diversity as long as possible) and so that the final adjustment of weights does not have (on average...) a major impact on `w_i` .

Runing the algorithm takes around `400 sec` on our computer and best final fit is around `-300` but after gradient descent on all `5,000` best points, we get a best fit around `-320` to be compared with the target `-330`. We can slightly improve this result by barycentration on best points. But what is really more important is that we have divided processing time by `10` with results similar to simple simulated annealing method and without parallelizing...

### test barycentric approach in dimension `500`
- gas has `cloud_size=50,000` particles as a tradeoff between diversity and computation capacity... Still it will be sparser than in dimension `50`.
- `max_run = 2,000` this parameter has to be increased so that particles can move in more directions (dimensionality issue)
- `accept=0.1` this value is on the fitness function and does not change much with dimensionality
- `min_main_weight = 0.9`, `neighbours=15` and `neg_margin=0.8` : We increase the number of neighbours to get more chances to test barycentration with all particles of the gas. We increase the negative margin because it is more difficult to explore out of the convex hull when the number of particles is big. 

With these parameters, computation takes `2 hours`, we get a best final fitness around `0` that can be decreased down to `-100` thanks to barycentration. Considering our computer's power, we take this as a good result.

### implementation with real data
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

Unfortunately a gradient descent on the `250,000 best points` is too costly. So we take a random sample of `5,000` to perform the gradient descent. On high dimensions, the gradient values may have a very wide range and it creates a risk of failure. We use a small `ro=0.005` (learning rate) and additionnaly clip any move in any direction in `[-0.2, 0.2]`. We run the algorithm in `5 parallel threads`. It takes `350 sec` to run. We see a good diversity of final fitness with a best one around `-230`.

We also try to take barycentres of 5 randomly chosen points in the gas. With `5,000 barycentres`, it takes more or less the same time to run (`350 sec`) for a similar best fitness. Both approaches appear to be receivable. 

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

I have read somewhere that Griewank's function is easier to optimize in high dimension. Based on that assumption, I use a very simple version of a generalized gradient descent algorithm. In a first computation, I implement a descent on cloud of points based on a gradient estimated with a very big step (`+/-5 on each coordinate`) and huge `ro` value, with almost no limitation on the size of the move. That way, I capture a global tendency of the function. Once this cloud is stabilized, I implement a classical approximated gradient descent on the barycentre of the cloud.

## Optimize shifted Griewank’s Function in dimension `50` with gradient descent

1st step :
- `cloud_size = 300`, it is a bit pessimistic but in dimension `50`, this precaution will not cost much.
- `eps = 5` is the size of the step to estimate the "gradient" (in this case, we call it a gradient because of the method but, of course, it has nothing to do with a mathematical gradient)
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



