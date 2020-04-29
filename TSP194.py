#**************
# TSP Qatar  *
#**************
print("TSP194")
# load libraries
import random
import numpy as np
from TSP_util import *
from matplotlib import pyplot as plt
import os

# random.seed(1203)
random.seed(1815)

# load raw data
quat = np.loadtxt("http://www.math.uwaterloo.ca/tsp/world/qa194.tsp", delimiter=" ", 
                  skiprows=7, max_rows=194)
quat = quat[:, 1:3]
n = len(quat)

# create problem
myTSP = TSP2(quat)

# get distances matrix
D = myTSP.matrix_of_distances

# visualize distribution of distances
myDist = np.sort( D, axis = None)[ n::2 ]
myHist = plt.hist( myDist, density=True)
plt.title(label = 'Distribution of city distances TSP 194')
plt.show()


#******************************************************************
#**************************GENETIC ALGORITHM***********************
#******************************************************************

print("\n Genetic Algorithm")

print("\n - using Hui's parameters with inverse proportional density for selection")
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta=0), 
       StoppingByEvaluations(max_evaluations=20000))

print("\n - with exponential density for selection")
# weight = 2 => exponentional density for selection
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0.001), 
       StoppingByEvaluations(max_evaluations=20000))

print("\n - reduce population and increase iterations")
tryTSP(myTSP, 100, 100, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta=0), 
       StoppingByEvaluations(max_evaluations=100000))

print("\n - increase mutation probability")
tryTSP(myTSP, 100, 100, 
       PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta=0), 
       StoppingByEvaluations(max_evaluations=100000))

print("\n - with exponential density for selection and less severe selection")
myResult = tryTSP(myTSP, 100, 100, 
                  PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
                  PMXCrossover(0.3), 
                  RouletteWheelSelection(beta = 0.0001), 
                  StoppingByEvaluations(max_evaluations=100000))

print("\n - reduce population and offspring")
myResult = tryTSP(myTSP, 30, 10, 
                  PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
                  PMXCrossover(0.3), 
                  RouletteWheelSelection(beta = 0.00005), 
                  StoppingByEvaluations(max_evaluations=100000))


#******************************************************************
#*******************CLOSEST NEIGHBOUR PATH*************************
#******************************************************************


print("\n Closest neighbour algorithm")

print("\n Find optimal path w.r.t. closest neighbour")
(totDist, optPath, arrDist) = findGLobalOptPath(D, n)
bestPath = np.argmin(totDist)
print("path length = ", totDist[bestPath])

# prepare for optimizing
sortedD = np.argsort(D)

print("\n - Optimize this path with best mutation")
(optPath2, dist2) =  optimizePath(D, n, optPath[ bestPath, : ], 
                                  verbal = False, maxLoop = 50)
print("path length =", dist2)


print("\n - with a randomly chosen mutation among the best ones")
(optPath3, dist3) =  optimizePath(D, n, optPath[ bestPath, : ], 
                                  verbal = False, maxLoop = 50)
print("path length =", dist3)

print("\n - with a randomly chosen improving mutation")
(optPath4, dist4) =  optimizePath(D, n, optPath[ bestPath, : ], 
                                  verbal = False, maxLoop = 50, first=True)
print("path length =", dist4)

input("Press enter to exit")