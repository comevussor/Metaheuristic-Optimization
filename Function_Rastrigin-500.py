#**************************************************************************************
#****************************RASTRIGIN's FUNCTION*************************************
#**************************************************************************************

#***************************in dimension 500*********************************************
print("Rastrigin in dimension 500 with real data")


import numpy as np
from joblib import Parallel, delayed
import random as rd
import time
# locals
import Functions_util2 as fs
import data


rd.seed(2610)
rng = np.random.default_rng(1611)

D = [50, 500]
lim = [-5, 5]
lD = len(D)
bias = -330

# get real shifts and shifted functions
x_shifts_r = [ np.array( data.rastrigin[0:d] ) for d in D]
s_fam_r = [ fs.define_rast_multi( x_shifts_r[k], bias ) for k in range(lD) ]

max_run = 50
reshuffle = 20
accept = 0.1
verbal = False
cloud_size = 50000
n_cloud = 5
n_jobs = 5
dim = D[1]
f = s_fam_r[1]

neg_margin = 0.8
neighbours = 15
min_main_weight = 0.9
    
XX_0 = np.random.random((n_cloud, cloud_size, dim)) * ( lim[1] - lim[0] ) + lim[0]

print("\n Perform parallelized barycentric approach of simulated annealing on 250,000 points")
print(" with 1,000 iterations, best fitness is printed every 50 iterations")

ff = fs.para_anneal_bary(dim, lim, f, max_run = max_run, accept=accept, neg_margin=neg_margin, 
                         neighbours=neighbours, min_main_weight=min_main_weight, 
                         verbal=False, order=True)

    
t0 = time.time()

for i in range( reshuffle ):

    my_results = Parallel( n_jobs=n_jobs )( delayed( ff )( XX_0[j] ) 
                                           for j in range( n_cloud ) )

    XX_min, YY_min, RRun_min = zip(*my_results)
    XX_min, YY_min, RRun_min = np.array(XX_min), np.array(YY_min), np.array(RRun_min)
    print(np.min(YY_min))
    X_min = XX_min[0]

    for j in range( n_cloud - 1 ):
        X_min = np.append( X_min, XX_min[j+1], axis=0 )

    haz = np.arange( n_cloud * cloud_size )
    np.random.shuffle(haz)
    X_min = X_min[haz]
    XX_0 = np.array( [ X_min[ ( j * cloud_size ):( ( j + 1 ) * cloud_size ) ] 
                      for j in range( n_cloud ) ] )

t1 = time.time()
t = t1-t0
print("Time to compute = ", t)

Y_min = YY_min[0]
for j in range( n_cloud - 1 ):
        Y_min = np.append(Y_min, YY_min[j+1])

# gradient descent
eps=0.01
ro=0.005
max_run=50
min_step=0
max_step=0.2
verbal=False

gg = fs.para_gradient(dim, lim, f, eps, ro, max_run=max_run, 
                      max_step=max_step, verbal=verbal)

n_jobs = 5
per_job = 100 
n_iter = 10

haz = np.arange( n_cloud * cloud_size )
np.random.shuffle(haz)
test = X_min[haz]

print("Perform gradient descent on random sample of 5,000 best points")

t0 = time.time()
results = Parallel(n_jobs=n_jobs)(delayed(gg)(test[ (per_job*i):(per_job*(i+1)) ]) 
                                  for i in range(n_iter))
t1 = time.time()
t = t1-t0
print(t)

XX_grad, YY_grad, RRuns_grad = zip(*results)
XX_grad, YY_grad, RRuns_grad = np.array(XX_grad), np.array(YY_grad), np.array(RRuns_grad)

print("Best fitness = ", np.min(YY_grad))

per_job = 200
bary_on_n = 5
n_jobs=5
n_picks = n_jobs * per_job
picks = np.random.randint(0, cloud_size * n_cloud, ( n_picks, bary_on_n ))
bary_list = np.array( [ np.mean( X_min[ picks[i] ], axis=0 ) 
                       for i in range( len(picks) ) ] )

print("Perform gradient descent on 5,000 barycentres of 5 randomly picked best points")

t0 = time.time()
results = Parallel( n_jobs=n_jobs )( delayed( gg )( bary_list[ (per_job * i):( per_job * ( i + 1 ) ) ]) 
                                    for i in range(n_jobs) )
t1 = time.time()
t = t1-t0
print(t)

XX_grad, YY_grad, RRuns_grad = zip(*results)
XX_grad, YY_grad, RRuns_grad = np.array(XX_grad), np.array(YY_grad), np.array(RRuns_grad)

print("Best fitness = ", np.min(YY_grad))

# input("Press enter to see error")

# bestXindex = np.unravel( np.argmin( YY_grad ), YY_grad.shape() )
# print("Final error = ", np.linalg.norm(x_shifts_r[1] - XX_grad[ bestXindex ] ) )

input("Press enter to exit")