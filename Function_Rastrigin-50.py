#**************************************************************************************
#****************************RASTRIGIN's FUNCTION*************************************
#**************************************************************************************
#***************************in dimension 50*********************************************
print("Rastrigin in dimension 50")

import numpy as np
import Functions_util2 as fs
from joblib import Parallel, delayed
import random as rd
import data
import time
from multiprocessing import Pool

rd.seed(2610)
rng = np.random.default_rng(1611) # insert seed here 1611 for reproducibility

D = [50, 500]
lim = [-5, 5]
lD = len(D)
bias = -330
n_try = 10

# generate random shifts and shifted functions
x_shifts = [fs.define_x_rand(count = n_try, dim = dim, lim = lim) for dim in D]
s_fam = [ fs.define_rast_multi(x_shifts[k][0], bias) for k in range(len(x_shifts))]

#********************** Test with random data******************************************

max_run = 10**6
divers = 1.5
intens = 10**-2
accept = 0.3
verbal = False
cloud_size = 10
dim = D[0]
f = s_fam[0]
n_jobs = -2 # all CPUs - 1
sample_size = 10

ff = fs.para_anneal(dim, lim, f, max_run=max_run, divers=divers, intens=intens, 
                    accept=accept, verbal=verbal)
XX_0 = np.random.random((sample_size, cloud_size, dim)) * ( lim[1] - lim[0] ) + lim[0]

print("\n Perform parallelized simulated annealing on random data starting with")
print(" 10 groups of 10 point")

t0 = time.time()
results = Parallel(n_jobs=n_jobs)(delayed(ff)(XX_0[i]) for i in range(sample_size))
t1 = time.time()

t = t1-t0
print(t)

XX_min, YY_min, RRuns = zip(*results)

XX_min, YY_min, RRuns = np.array(XX_min), np.array(YY_min), np.array(RRuns)

print("fitnesses = ", YY_min)
print("number of runs = ", RRuns)

print("Best fitness = ", np.min(YY_min))

eps = 0.01
ro = 0.1
max_run = 80
max_step = 0.01
verbal = False

gg = fs.para_gradient(dim, lim, f, eps, ro, max_run=max_run, max_step=max_step, 
                      verbal=verbal)

print("\n Perform gradient descent on all the best points resulting from annealing")

t0 = time.time()
results = Parallel(n_jobs=n_jobs)(delayed(gg)(XX_min[i]) for i in range(sample_size))
t1 = time.time()

t = t1-t0
print("Computation time = ", t)

XX_grad, YY_grad, RRuns_grad = zip(*results)
XX_grad, YY_grad, RRuns_grad = np.array(XX_grad), np.array(YY_grad), np.array(RRuns_grad)

print("Fitnesses = ", YY_grad)

# take barycenter of points in the cloud

print("\n Perform gradient descent on the barycenters of groups of") 
print("  10 best points from annealing")

barycenter = np.mean(XX_grad, axis=1)
barycenter_fitness = fs.gradient_descent(dim, lim, barycenter, f, eps=eps, ro=ro, 
                                         max_run=5*max_run, max_step=max_step, 
                                         verbal=False)

print("Final fitnesses = ", barycenter_fitness[1])
print(" after ", barycenter_fitness[2], " runs")

print("best fitness = ", np.min(barycenter_fitness[1]))

#********************* Implement with real data in dimension 50 ***********************


# Load functions

D = [50, 500]
lim = [-5, 5]
lD = len(D)
bias = -330
x_shifts_r = [ np.array( data.rastrigin[0:d] ) for d in D]
s_fam_r = [ fs.define_rast_multi( x_shifts_r[k], bias ) for k in range(lD) ]


max_run = 10**6
divers = 1.5
intens = 10**-2
accept = 0.3
verbal = False
dim = D[0]
f = s_fam_r[0]
# we pick sample_size samples of cloud_size points and run all CPUs -1 in parallel
cloud_size = 10
sample_size = 50

print("\n Perform parallelized simulated annealing on real data (50 groups of 10 points)")

ff = fs.para_anneal(dim, lim, f, max_run=max_run, divers=divers, intens=intens, 
                    accept=accept, verbal=verbal)
XX_0 = np.random.random((sample_size, cloud_size, dim)) * ( lim[1] - lim[0] ) + lim[0]

t0 = time.time()
results = Parallel(n_jobs=n_jobs)(delayed(ff)(XX_0[i]) for i in range(sample_size))
t1 = time.time()

t = t1-t0
print(t)

XX_min, YY_min, RRuns = zip(*results)

XX_min, YY_min, RRuns = np.array(XX_min), np.array(YY_min), np.array(RRuns)

print("YY_min = ", YY_min)
print("RRuns = ", RRuns)

print(np.min(YY_min))

eps = 0.01
ro = 0.1
max_run = 500
max_step = 0.01
verbal = False

gg = fs.para_gradient(dim, lim, f, eps, ro, max_run=max_run, max_step=max_step, 
                      verbal=verbal)

print("\n Perform gradient descent on final points")

t0 = time.time()
results = Parallel(n_jobs=n_jobs)(delayed(gg)(XX_min[i]) for i in range(sample_size))
t1 = time.time()

t = t1-t0
print(t)

XX_grad, YY_grad, RRuns_grad = zip(*results)
XX_grad, YY_grad, RRuns_grad = np.array(XX_grad), np.array(YY_grad), np.array(RRuns_grad)
# XX_grad shape : (50, 10, 50)

print(YY_grad)
print(np.min(YY_grad))

# take barycenters of each sample of 10 points
barycenter = np.mean(XX_grad, axis=1)
# barycenter shape (50,50)

print("\n Perform gradient descent on the barycenters of groups of") 
print("  10 best points from annealing")

barycenter_fitness = fs.gradient_descent(dim, lim, barycenter, f, eps=eps, ro=ro, 
                                         max_run=5*max_run, max_step=max_step, 
                                         verbal=False)
print("Fitnesses = ", barycenter_fitness[1])
print("Best fitness = ", np.min(barycenter_fitness[1]))
print("Error = ", 
      np.linalg.norm(x_shifts_r[0]- \
          barycenter_fitness[0][np.argmin(barycenter_fitness[1])]))


input("Press enter to exit")