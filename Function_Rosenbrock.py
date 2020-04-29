#**************************************************************************************
#****************************ROSENBROCK's FUNCTION*************************************
#**************************************************************************************
print("Rosenbrock")

import psopy as pso
import scipy.optimize as opt
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt 
import Functions_util as fs
import random as rd
import data
import time

rd.seed(2610)

D = [50, 500]
L = [-100, 100]
lD = len(D)
Dmax = max(D)
n_try = 100

# generate random shifts and shifted functions
x_shifts = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_family_rosen(x_shifts[k]) for k in range(len(x_shifts))]

# with real data
x_shifts_r = [ np.array( data.rosenbrock[0:d] ) for d in D]
s_fam_r = [ fs.define_rosen( dim = D[k], x_shift = x_shifts_r[k] ) for k in range(lD) ]

# generate random initial values for x
x_0s = [fs.define_x_rand(count = n_try, dim = d) for d in D]

# **************************************in dimension 50*********************************

print("\n Simulated annealing followed by gradient descent in dimension 50")

n_try = 5
d = 0
dim = D[d]

#*************with random data

print("\n With random data")

# simulated annealing

print("\n - simulated annealing : 5 successive trials ")
maxIter = 5*10**5
divers = 0.3
intens = 10 ** -3 
verbal = 2

start = time.time()
myResults = [ fs.simulateAnnealing2(s_fam[d][i], x_0s[d][i], maxIter=maxIter,
                                    divers=divers, intens=intens, verbal=verbal)
              for i in range(n_try) ]
end = time.time()

avgTime = (end - start) / n_try
print('average time 1 trial =', avgTime)

# *******************with real data

print("\n With real data")

# simulated annealing

print("\n - simulated annealing : 5 successive trials ")

maxIter = 5*10**5
divers = 0.3
intens = 10 ** -3 
verbal = 2

start = time.time()
myResults_r = [fs.simulateAnnealing2(s_fam_r[d], x_0s[d][i], maxIter=maxIter,
                                    divers=divers, intens=intens, verbal=verbal)
              for i in range(n_try) ]
end = time.time()
avgTime = (end - start) / n_try

print('average time 1 trial =', avgTime)

# local search with gradient

print("\n - gradient descent on best point ")

myVal = [myResults_r[i]["minimum_energy"] for i in range(n_try)]
iBest = np.argmin(myVal)
x0 = myResults_r[iBest]["minimum_state"]

start = time.time()
xl, fl, nl = fs.gradient_method(dim, x0, s_fam_r[d], eps=10**-3, ro=10**-3, 
                                termination_criterion=fs.termination_by_runs,
                                max_run = 6000, min_step = 1, bounds = [-100, 100], 
                                ro_adjust = False, ro_optim = True, eps_adjust = False)
end = time.time()

print("gradient time : ", end-start)
print("fitness = ", fl," after ", nl, " runs")
print("Error = ", np.linalg.norm(xl-x_shifts_r[d]))

# ************************in dimension 500*************************************

print("\n Simulated annealing followed by gradient descent in dimension 500")

n_try = 5
d = 1
dim = D[d]

#**************with random data

print("\n With random data")

# simulated annealing 

print("\n - simulated annealing : 5 successive trials (be patient...)")

maxIter = 3*10**6
divers = 0.3
intens = 10 ** -4
verbal = 2
accept = 500

start = time.time()
myResults = [ fs.simulateAnnealing2(s_fam[d][i], x_0s[d][i], maxIter=maxIter,
                                    divers=divers, intens=intens, accept=accept,
                                    verbal=verbal)
              for i in range(n_try) ]
end = time.time()
avgTime = (end - start) / n_try

print('average time =', avgTime)


myVal = [myResults[i]["minimum_energy"] for i in range(n_try)]
iBest = np.argmin(myVal)
x0 = myResults[iBest]["minimum_state"]
print("The difference between the target shift and the shift we found is : ")
print(x0-x_shifts[1][iBest])


# local search with gradient

print("\n - gradient descent on best point ")

start = time.time()
xl, fl, nl = fs.gradient_method(dim, x0, s_fam[d][iBest], eps=10**-5, ro=10**-4, 
                             termination_criterion=fs.termination_by_runs,
                             max_run = 2000, min_step = 1, bounds = [-100, 100], 
                             ro_adjust = False, ro_optim = True, eps_adjust = False)
end = time.time()

print("gradient time : ", end-start)

print("fitness = ", fl," after ", nl, " runs")
print("Error = ", np.linalg.norm(xl-x_shifts[1][iBest]))

# *****************with real data

print("\n With real data")

# simulated annealing

print("\n - simulated annealing : 5 successive trials ")

maxIter = 3*10**6
divers = 0.3
intens = 10 ** -4
verbal = 2
accept = 500
n_try = 5

start = time.time()
myResults_r = [fs.simulateAnnealing2(s_fam_r[d], x_0s[d][i], maxIter=maxIter,
                                     divers=divers, intens=intens, accept=accept,
                                     verbal=verbal)
              for i in range(n_try) ]
end = time.time()
avgTime = (end - start) / n_try

print('average time 1 trial =', avgTime)

# local search with gradient

print("\n - gradient descent on best point ")

myVal = [myResults_r[i]["minimum_energy"] for i in range(n_try)]
iBest = np.argmin(myVal)
x0 = myResults_r[iBest]["minimum_state"]

start = time.time()
xl, fl, nl = fs.gradient_method(dim, x0, s_fam_r[d], eps=10**-3, ro=10**-3, 
                                termination_criterion=fs.termination_by_runs,
                                max_run = 5000, min_step = 1, bounds = [-100, 100], 
                                ro_adjust = True, ro_optim = True, eps_adjust = True)
end = time.time()

print("gradient time : ", end-start)
print("fitness = ", fl," after ", nl, " runs")

print("Error = ", np.linalg.norm(xl-x_shifts_r[d]))

input("Press enter to exit")