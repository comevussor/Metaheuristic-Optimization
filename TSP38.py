#****************
# TSP DJIBOUTI  *
#****************
print("TSP38")

# load librarys
import random
import numpy as np
from TSP_util import *
from matplotlib import pyplot as plt
import time

# reproducibility
random.seed(1203)
# random.seed(1815)

# load raw data
djib = np.loadtxt( "http://www.math.uwaterloo.ca/tsp/world/dj38.tsp", 
                  delimiter=" ", skiprows=10 )
djib = djib[ :, 1:3 ]
n = len( djib )

# create model
myTSP = TSP2(djib)

# compute distances and visualize distances distribution
D = myTSP.matrix_of_distances
myDist = np.sort( D, axis = None)[ n::2 ]

myHist = plt.hist( myDist, density=True)
plt.title(label = 'Distribution of city distances TSP 38')
plt.show()

#******************************************************************
#**************************GENETIC ALGORITHM***********************
#******************************************************************

print("\n Genetic Algorithm")

print("\n - using Hui's parameters with inverse proportional density for selection")
# first try with Hui parameters except population because of computation limitation
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=1), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=20000))

print("\n - with exponential density for selection")
# beta = 0.001 => exponentional density for selection
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=1), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0.001), 
       StoppingByEvaluations(max_evaluations=20000))

print("\n - with optimal mutation and inverse proportional density for selection")
# randMut = 2 => optimal mutation
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=20000))

print("\n - with optimal mutation and inverse exponential density for selection")
# beta=0.001 => exponentional density for selection
tryTSP(myTSP, 1000, 1000, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=20000))


print("\n - reduce population and increase iterations")
tryTSP(myTSP, 100, 100, 
       PermutationSwapMutation(0.01, randMut=2, D=D, n=n, first=False), 
       PMXCrossover(0.3), 
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=100000))

print("\n - increase mutation probability")
tryTSP(myTSP, 100, 100,
       PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False),
       PMXCrossover(0.3),
       RouletteWheelSelection(beta = 0), 
       StoppingByEvaluations(max_evaluations=100000))

# Iterate the elected method

print("\n The next computation is lengthy, (100 implementations of the previous one)")
YN = input("Do you want to implement it ?")

if YN == "Y" :
    fitnesses = np.zeros(100)
    for i in range(100):
        r = tryTSP(myTSP, 100, 100, 
                    PermutationSwapMutation(0.5, randMut=2, D=D, n=n, first=False), 
                    PMXCrossover(0.3), 
                    RouletteWheelSelection(beta = 0), 
                    StoppingByEvaluations(max_evaluations=100000))
        fitnesses[i] = r.objectives[0]

    # get minimum from sample
    print("\n Best fitness = ", np.min(fitnesses))
else :
    print("\n Understood, we pursue")

#******************************************************************
#*******************CLOSEST NEIGHBOUR PATH*************************
#******************************************************************

print("\n Closest neighbour algorithm")

print("\n Find optimal path w.r.t. closest neighbour")
(totDist, optPath, arrDist) = findGLobalOptPath(D, n)
bestPath = np.argmin(totDist)
print("path length = ", totDist[bestPath])

print("\n Find optimal path w.r.t. closest neighbour and optimize with best mutation")
(totDist2, optPath2, arrDist1) = findGLobalOptPath(D, n, optimize=True, maxLoop=30)
bestPath2 = np.argmin(totDist2)
print("path length = ", totDist2[bestPath2])

input("Press enter to exit")