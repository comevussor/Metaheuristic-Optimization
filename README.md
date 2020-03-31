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

In dimension `50`, we start with random shifts to test different methods. We perform `100` runs of simulated annealing limited to `3.10^6` iterations, with initial step set at `1` and final step set at `5.10^-3` (we start with a global search and finish with a local search). We set `accept = 1` which means, according the function range, that we refuse almost any increase at the beginning and start tolerating increase much later, when step is way smaller. We get a cloud of points that we expect to be around the global minimum. Each run takes `2 min` on my computer. And we get minimum energy list :  
`[-120.16358571496738, -75.78344978772745, -69.05819743302106, -120.51101265994541, 15.850243165490951, -66.2992038050138, -118.76359419663757, -196.9227649160092, -131.98193670417604, -165.69228836145305, -80.01258139949391, -45.684349457413475, -113.26725434161136, -156.24593316589466, -138.37596488237168, -41.49974079854729, -50.842659537165446, 7.122330650211495, 1.1982858400099303, -94.64811150435241, -107.60280654777571, -15.652510824884416, -166.83432190723494, -101.86074842343703, -166.92730272418243, 54.95834131861625, -28.43493516330659, -105.26553842527272, -96.38425801314699, -81.73791181375472, 22.26141285574232, -146.21819635682647, -100.01965261103479, 22.856666260565703, -64.56363810803634, -83.4438617672198, -149.91873449004484, -100.85564763041052, -181.40944279330145, -84.4457826693125, -74.09627923528424, -85.22071155078743, 149.07260173404165, -110.05082606144359, -179.80777710804023, -33.798410152110876, -137.3686507818592, -74.42454348687983, -141.30564747871747, -173.04640296607565, -25.289247632869035, -129.9339705162757, -152.95586834673563, -79.80685419052338, -20.453974745029598, -98.0886910321504, -147.5991437466542, -105.51708167147859, -111.81866710129668, 36.63819941093897, -55.43472810316973, 67.61374900652487, -100.61266339572802, -133.81846229318103, -66.75620385748891, -49.63335141600646, 59.5390907990888, -127.46016953424524, -115.85399817147305, -163.61966908437023, -175.7117622454923, -142.53627718664282, -44.98603143687046, -81.39388816472459, -148.84293506098905, 49.27503594444988, -44.354321588862604, -87.50630063035277, -109.39516178315787, -47.13262868381497, -112.19242423930436, 90.72192121182047, -103.78234078929322, -108.54196427394425, 124.234008601776, 95.2554398286054, -182.88180737802725, -112.45037145686592, -133.7115663824237, -152.3316942271468, -77.95316528692845, -104.54069275913022, -84.5147276219354, -33.4770048346881, 73.33170536218938, -108.88016222049274, -195.92109914378713, -206.1068712304754, -104.38088817832252, -65.47622972697161]`

We run the gradient algorithm on each one of the corresponding best found state. The best result yields a fitness of `-213` to be compared with the target at `-330`. Each run take `2 sec`.  
We try to improve this result performing the gradient descent on the barycentre of the 100 points previously found. And indeed, we find a fitness at `-264` which is much better.  
Now we try to make a local search in the area of the cloud found thanks to simulated annealing + gradient method. We take random points computed as a weighted sum of the points in the cloud. We limit the weights to be mainly positive to stay in the area of the cloud. We take random weights ranging from `-0.7` to `1`. We choose `100` random points `400` times and we perform gradient method on each one of these `40,000` points. Fitness comes to `-289` within `1,500` seconds.  
Now, we use directly the points from simulated annealing without gradient descent and use the same system of random weighted points, an identical number of times (`40,000`), fitness comes to `-290`, meaning that gradient optimization on the cloud given by simulated annealing is useless.

As a preliminary conclusion, this method sounds interesting but 







