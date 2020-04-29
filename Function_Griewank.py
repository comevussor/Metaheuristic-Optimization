#**************************************************************************************
#****************************GRIEWANK's FUNCTION*************************************
#**************************************************************************************

print("Griewank function")

import numpy as np
import Functions_util2 as fs
from joblib import Parallel, delayed
import random as rd
import data
import time
from multiprocessing import Pool

# rd.seed(2610) 
rng = np.random.default_rng() # insert seed here 1611 for reproducibility

D = [50, 500]
lim = [-600, 600]
lD = len(D)
bias = -180
n_try = 10

# generate random shifts and shifted functions
x_shifts = [fs.define_x_rand(count = n_try, dim = dim, lim = lim) for dim in D]
s_fam = [ fs.define_griewank_multi(x_shifts[k][0], bias) for k in range(len(x_shifts))]

# get real shifts and shifted functions
x_shifts_r = [ np.array( data.griewank[0:d] ) for d in D]
s_fam_r = [ fs.define_griewank_multi( x_shifts_r[k], bias ) for k in range(lD) ]

# test 2 steps gradient method in dimension 50

print("\n in dimension 50 with random data :")

dim = D[0]
f = s_fam[0]

cloud_size = 300
eps=5
ro=100
max_run=200
min_step=0
max_step=30
verbal=False
terminate_by='runs'
X_0 = np.random.random((cloud_size, dim)) * ( lim[1] - lim[0] ) + lim[0]

print("\n Perform a pseudo-gradient descent on 300 points")

t0 = time.time()
test1 = fs.gradient_descent(dim, lim, X_0, f, eps, ro, terminate_by, max_run, 
                            min_step, max_step, verbal)
t1 = time.time()
t = t1-t0
print("Time to compute = ", t)

print("Best Fitness = ", np.min(test1[1]))

X_1 = test1[0]

xg_1 = np.mean(X_1, axis=0)
Xg = [xg_1]

eps=0.01
ro=1
max_run=500
min_step=0
max_step=1
verbal=False
terminate_by='runs'

print("\n Perform an estimated gradient descent on 300 resulting points")

test3 = fs.gradient_descent(dim, lim, Xg, f, eps, ro, terminate_by, max_run, 
                            min_step, max_step, verbal)

print("Fitness = ", test3[1])
print("Error = ", np.linalg.norm(test3[0][0] - x_shifts[0][0]))


# with real data in dimension 50

print("\n in dimension 50 with real data :")

dim = D[0]
f = s_fam_r[0]

cloud_size = 300
eps=5
ro=100
max_run=200
min_step=0
max_step=30
verbal=False
terminate_by='runs'
X_0 = np.random.random((cloud_size, dim)) * ( lim[1] - lim[0] ) + lim[0]

print("\n Perform a pseudo-gradient descent on 300 points")

t0 = time.time()
test1 = fs.gradient_descent(dim, lim, X_0, f, eps, ro, terminate_by, max_run, 
                            min_step, max_step, verbal)
t1 = time.time()
t = t1-t0
print("Time to compute = ", t)

print("Best Fitness = ", np.min(test1[1]))
X_1 = test1[0]

xg_1 = np.mean(X_1, axis=0)
Xg = [xg_1]

eps=0.01
ro=1
max_run=500
min_step=0
max_step=1
verbal=False
terminate_by='runs'

print("\n Perform an estimated gradient descent on 300 resulting points")

test3 = fs.gradient_descent(dim, lim, Xg, f, eps, ro, terminate_by, max_run, min_step, max_step, verbal)

print("Fitness = ", test3[1])
print("Error = ", np.linalg.norm(test3[0][0] - x_shifts_r[0]))

# test 2 steps gradient method in dimension 500 with random data

print("\n in dimension 500 with random data :")

dim = D[1]
f = s_fam[1]

cloud_size = 200
X_0 = np.random.random((cloud_size, dim)) * ( lim[1] - lim[0] ) + lim[0]

eps=5
ro=1000
max_run=50
min_step=0
max_step=300
verbal=False
terminate_by='runs'

print("\n Perform a pseudo-gradient descent on 200 points")

t0 = time.time()
test1 = fs.gradient_descent(dim, lim, X_0, f, eps, ro, terminate_by, max_run, min_step, 
                            max_step, verbal)
t1 = time.time()
t = t1-t0
print("Time to compute = ", t)

print("Fitnesses = ", test1[1])

X_1 = test1[0]
y_min = np.min(test1[1])
x_min = X_1[np.argmin(test1[1])]

print("Best fitness = ", y_min)
print("Error = ", np.linalg.norm(x_min - x_shifts[1][0]))

xg_1 = np.mean(X_1[ np.argsort( test1[1] )[1:100] ], axis=0)
Xg = [xg_1]

eps=0.01
ro=1
max_run=100
min_step=0
max_step=2
verbal=False
terminate_by='runs'

print("\n Perform an estimated gradient descent on barycentre of 100 best resulting points")

t0 = time.time()
test2 = fs.gradient_descent(dim, lim, Xg, f, eps, ro, terminate_by, max_run, 
                            min_step, max_step, verbal)
t1 = time.time()
t = t1-t0
print("Time to compute = ", t)

print("Fitness = ", test2[1])
print("Error = ", np.linalg.norm(test2[0][0] - x_shifts[1][0]))

# with real data in dimension 500

print("\n in dimension 500 with real data :")

dim = D[1]
f = s_fam_r[1]

cloud_size = 200
X_0 = np.random.random((cloud_size, dim)) * ( lim[1] - lim[0] ) + lim[0]

eps=5
ro=1000
max_run=50
min_step=0
max_step=300
verbal=False
terminate_by='runs'

print("\n Perform a pseudo-gradient descent on 200 points")

t0 = time.time()
test1 = fs.gradient_descent(dim, lim, X_0, f, eps, ro, terminate_by, max_run, min_step, 
                            max_step, verbal)
t1 = time.time()
t = t1-t0
print("Time to compute = ", t)

X_1 = test1[0]
y_min = np.min(test1[1])
x_min = X_1[np.argmin(test1[1])]

print("Best fitness = ", y_min)
print("Error = ", np.linalg.norm(x_min - x_shifts_r[1]))

xg_1 = np.mean(X_1[ np.argsort( test1[1] )[1:100] ], axis=0)
Xg = [xg_1]

eps=0.01
ro=1
max_run=100
min_step=0
max_step=2
verbal=False
terminate_by='runs'

print("\n Perform an estimated gradient descent on barycentre of 100 best resulting points")

t0 = time.time()
test2 = fs.gradient_descent(dim, lim, Xg, f, eps, ro, terminate_by, max_run, 
                            min_step, max_step, verbal)
t1 = time.time()
t = t1-t0
print("Time to compute = ", t)

print("Fitness = ", test2[1])
print("Error = ", np.linalg.norm(test2[0][0] - x_shifts_r[1]))

input("Press enter to exit")