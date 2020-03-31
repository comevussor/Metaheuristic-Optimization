# load tools
import random
import numpy as np
from TSP_util import *
import os

# random.seed(1203)
random.seed(1815)

#**************
# TSP Quatar  *
#**************

# load raw data
quat = np.loadtxt("http://www.math.uwaterloo.ca/tsp/world/qa194.tsp", delimiter=" ", skiprows=7, max_rows=194)
quat = quat[:, 1:3]
print(quat)
n = len(quat)

# create problem
myTSP = TSP2(quat)

# check global bounds
D = myTSP.matrix_of_distances
print(np.sort(193*D, axis = None)[193:])

# First try with Hui parameters

tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta=0), 
       StoppingByEvaluations(max_evaluations=20000))

# weight = 2 => exponentional density for selection
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0.001), 
       StoppingByEvaluations(max_evaluations=20000))

# reduce population, increase iterations
tryTSP(myTSP, 100, 100, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta=0), 
       StoppingByEvaluations(max_evaluations=100000))

# same with increase in mutation probability
tryTSP(myTSP, 100, 100, 
       PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta=0), 
       StoppingByEvaluations(max_evaluations=100000))

# increase in mutation probability + exponential density for selection with less severe selection
myResult = tryTSP(myTSP, 100, 100, 
                  PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
                  PMXCrossover(0.3), 
                  RouletteWheelSelection(beta = 0.0001), 
                  StoppingByEvaluations(max_evaluations=100000))

myResult = tryTSP(myTSP, 30, 10, 
                  PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
                  PMXCrossover(0.3), 
                  RouletteWheelSelection(beta = 0.00005), 
                  StoppingByEvaluations(max_evaluations=100000))

# find closest neighbour path
# remember that distance matrix is D

# find best neighbour matrix
(totDist, optPath, arrDist) = findGLobalOptPath(D, n)
print(totDist, optPath, arrDist)

bestPath = np.argmin(totDist)
print(totDist[bestPath])

# prepare for optimizing
sortedD = np.argsort(D)

# optimize shortest of shortest
# with best mutation
(optPath2, dist2) =  optimizePath(D, n, optPath[ bestPath, : ], verbal = False, maxLoop = 50)
print(optPath2, dist2)

# with a randomly chosen mutation among the best
(optPath3, dist3) =  optimizePath(D, n, optPath[ bestPath, : ], verbal = False, maxLoop = 50)
print(optPath3, dist3)

# with a random improving mutation
(optPath4, dist4) =  optimizePath(D, n, optPath[ bestPath, : ], verbal = False, maxLoop = 50, first=True)
print(optPath4, dist4)
