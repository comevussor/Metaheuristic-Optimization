# load tools
import random
import numpy as np
from TSP_util import *

# reproducibility
random.seed(1203)
# random.seed(1815)

#****************
# TSP DJIBOUTI  *
#****************

# load raw data
djib = np.loadtxt("http://www.math.uwaterloo.ca/tsp/world/dj38.tsp", delimiter=" ", skiprows=10)
djib = djib[:, 1:3]
print(djib)
n = len(djib)

# indices = np.arange(38)
# np.random.shuffle(indices)
# djib = djib[indices]

# create problem
myTSP = TSP2(djib)

# check rough global bounds
D = myTSP.matrix_of_distances
print(np.sort(38*D, axis = None)[36:])

# scale roulette exponential weighting (see RouletteWheelSelection)

# first try with Hui parameters except population because of computation limitation
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=1), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=20000))

# beta = 0.001 => exponentional density for selection
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=1), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0.001), 
       StoppingByEvaluations(max_evaluations=20000))

# randMut = 2 => optimal mutation
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=20000))

# beta=0.001 => exponentional density for selection
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=20000))

# reduce population and increase iterations
tryTSP(myTSP, 100, 100, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=100000))

tryTSP(myTSP, 100, 100, 
                  PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
                  PMXCrossover(0.3), 
                  RouletteWheelSelection(beta = 0), 
                  StoppingByEvaluations(max_evaluations=100000))

# sample last method
fitnesses = np.zeros(100)
for i in range(100):
    r = tryTSP(myTSP, 100, 100, 
               PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
               PMXCrossover(0.3), 
               RouletteWheelSelection(beta = 0), 
               StoppingByEvaluations(max_evaluations=100000))
    fitnesses[i] = r.objectives[0]

# get minimum from sample
print(fitnesses)
print(np.min(fitnesses))

# find closest neighbour path
# remember that distance matrix is D, get its max

# find best neighbour
(totDist, optPath, arrDist) = findGLobalOptPath(D, n)
print(totDist, optPath, arrDist)
bestPath = np.argmin(totDist)
print(totDist[bestPath])
print(optPath[bestPath])

# find best optimized neighbour
(totDist2, optPath2, arrDist1) = findGLobalOptPath(D, n, optimize=True, maxLoop=30)
print(totDist2, optPath2)
bestPath2 = np.argmin(totDist2)
print(totDist2[bestPath2])

# find almost best (randomize mutation) neighbour and optimize
(totDist3, optPath3, arrDist1) = findGLobalOptPath(D, n, optimize=True, maxLoop=30)
print(totDist3, optPath3)
bestPath3 = np.argmin(totDist3)
print(totDist3[bestPath3])
