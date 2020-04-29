#***********************************************************************************
#******************************SPHERE FUNCTION*************************************
#***********************************************************************************
print("sphere")

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
n_try = 25
lD = len(D)
Dmax = max(D)

# generate random shifts, and initial values of x and functions
x_shifts = [fs.define_x_rand(count = n_try, dim = d) for d in D]
x_0s = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_family_sphere(x_shifts[k]) for k in range(len(x_shifts))]

#***********************************************************************************
#******************************GRADIENT DESCENT*************************************
#***********************************************************************************

print("\n Estimated gradient descent with random shift")

print(
    "\n Descent is optimized after each gradient computation \
    \n Learning rate is adjusted along the way")

eps = 10**(-4) 
ro = 0.5
verbal = False
ro_adjust = True
ro_optim = True 
max_run = 100
# with random data

x_opts = [np.empty((n_try, D[i]), dtype=float) for i in range(lD)]
f_opts = np.empty((lD, n_try), dtype=float)
n_runs = np.empty((lD, n_try), dtype=float)

print("\n Algorithm is implemented 25 times in each dimension on different shifts")
for i in range(lD):
    dim = D[i]
    min_step = dim * 10**(-4)
    for t in range(n_try):
        print("dimension = ", dim, " ; try #", t)
        x_0 = x_0s[i][t]
        f = s_fam[i][t]
        (x_opts[i][t], f_opts[i][t], n_runs[i][t]) \
            = fs.gradient_method(dim, x_0, f, eps=eps, ro=ro, ro_optim=ro_optim, ro_adjust=ro_adjust,
                                 termination_criterion = fs.termination_by_step, max_run=max_run,
                                 min_step = min_step, verbal=verbal)

print("\n final fitnesses :")
print(f_opts)
print("\n number or runs")
print(n_runs)

print("\n Estimated gradient descent with real shift in dimension 50 and 500")
x_shifts = [ np.array(data.sphere[0:d]) for d in D]
x_0s = [fs.define_x_rand(count = 2, dim = d) for d in D]
s_fam = [fs.define_sphere(dim = D[k], x_shift = x_shifts[k]) for k in range(lD)]
x_opts = [np.empty((1, D[i]), dtype=float) for i in range(lD)]
f_opts = np.empty((lD, 1), dtype=float)
n_runs = np.empty((lD, 1), dtype=float)

for i in range(lD):
    dim = D[i]
    min_step = dim * 10**(-4)
    f = s_fam[i]
    x_0 = x_0s[i][0]
    (x_opts[i][0], f_opts[i][0], n_runs[i][0]) \
        = fs.gradient_method(dim, x_0, f, eps=eps, ro=ro, ro_optim=ro_optim, ro_adjust=ro_adjust,
                                termination_criterion = fs.termination_by_step, max_run=max_run,
                                min_step = min_step, verbal=verbal)

print("fitnesses = ", f_opts)
print("number of runs", n_runs)
print("Errors :")
print( np.linalg.norm( x_opts[0][0] - x_shifts[0]) )
print( np.linalg.norm( x_opts[1][0] - x_shifts[1]) )

#***********************************************************************************
#******************************PSO**************************************************
#***********************************************************************************


#********************************PSO in dimension 50********************************
print("\n PSO algorithm with random data in dimension 50")

# set constraints
const = [{'type':'ineq', 'fun': lambda x: x[i] - L[0]} for i in range(Dmax) ]
const.extend( [ {'type':'ineq', 'fun': lambda x: -x[i] + L[1]} for i in range(Dmax) ] )

# generate random shifts, and initial values of x and functions
n_pop = 50
n_try = 200

x_shifts = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_family_sphere(x_shifts[k]) for k in range(lD)]

# test learning rates
rmin = 0.5
rmax = 5.5
ncheck = 11
step = (rmax - rmin) / (ncheck - 1)
rates = np.empty((ncheck, ncheck))
values = np.arange(rmin, rmax+0.1, step)

print("\n Try different values of learning rates in dimension 50 :")
g = 0
for grate in values:
    l = 0
    for lrate in values:
        x_0s = [fs.define_x_rand(count = n_pop, dim = d) for d in D]
        i = (ncheck * l + g) % 25
        f = s_fam[0][i]
        myResult = pso.minimize(f, x_0s[0], 
                                options = {'max_iter':200,
                                          'verbose':False,
                                          'friction':0.5,
                                          'g_rate':grate,
                                          'l_rate':lrate},
                               constraints = const[0:50].extend(const[500:550]))
        rates[l, g] = myResult.fun
        print("local rate = ", lrate,
             " ; global rate = ", grate,
            " ; fitness = ", rates[l, g])
        l += 1
    g += 1

rates1 = rates
plt.imshow(rates1, cmap=cm.gray)
plt.title(label = "Fitness of PSO according to learning rate \n \
            (global_rate*2, local_rate*2) \
            \n in dimension 50")
plt.show()

print("\n Implement PSO with real data, still in dimension 50")

n_pop = 500
n_try = 1
max_iter = 1000
g_rate = 5
l_rate = 0.2
friction = 0.5
max_velocity = 5
x_shifts = [ np.array( data.sphere[0:d] ) for d in D]
s_fam = [ fs.define_sphere( dim = D[k], x_shift = x_shifts[k] ) for k in range(lD) ]

dim = 50
start = time.time()
for t in range(n_try):
    x_0s = fs.define_x_rand(count = n_pop, dim = dim)
    myResults = np.empty((25, 5), dtype = float)
    myXs = np.empty((25, dim), dtype=float)
    myResult = pso.minimize(s_fam[0], x_0s,
                            options = {'max_iter':max_iter,
                                       'verbose':False,
                                       'friction':friction,
                                       'max_velocity': max_velocity,
                                       'g_rate':g_rate,
                                       'l_rate':l_rate},
                            constraints = const[0:50].extend(const[500:550]))
    myResults[t,0] = myResult.fun
    myResults[t,1] = myResult.nit
    myResults[t,2] = myResult.nsit
    myResults[t,3] = myResult.status
    myResults[t,4] = myResult.success
    myXs[t] = myResult['x']
end = time.time()

# see results
print("fitness = ", myResults[:,0][0])
avgTime = (end - start)
print('time to compute = ', avgTime)


#********************************PSO in dimension 500********************************
print("\n PSO algorithm with random data in dimension 500")

# test learning rates

n_pop = 50
n_try = 200

x_shifts = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_family_sphere(x_shifts[k]) for k in range(lD)]

rmin = 0.5
rmax = 5.5
ncheck = 11
step = (rmax - rmin) / (ncheck - 1)
rates = np.empty((ncheck, ncheck))
values = np.arange(rmin, rmax+0.1, step)
valuesg = 2 * values

print("\n Try different values of learning rate in dimension 500 :")

g = 0
for grate in valuesg :
    l = 0
    for lrate in values :
        x_0s = [fs.define_x_rand(count = n_pop, dim = d) for d in D]
        i = (ncheck * l + g) % 25
        f = s_fam[1][i]
        myResult = pso.minimize(f, x_0s[1], 
                                options = {'max_iter':200,
                                          'verbose':False,
                                          'friction':0.5,
                                          'g_rate':grate,
                                          'l_rate':lrate})
#                               constraints = const)
        rates[l, g] = myResult.fun
        print("local rate = ", lrate,
             " ; global rate = ", grate,
            " ; fitness = ", rates[l, g])
        l += 1
    g += 1

rates2 = rates
plt.imshow(rates2, cmap=cm.gray)
plt.title(label = "Fitness of PSO according to learning rate \n \
            (global_rate, local_rate*2) \
            \n in dimension 500")
plt.show()


print("\n Implement PSO with real data, still in dimension 500")

n_pop = 1000
n_try = 1
max_iter = 1500
g_rate = 7
l_rate = 0.2
friction = 0.9
max_velocity = 1

# real data
# we penalize the objective function to avoid checking constraints
x_shifts = [ np.array( data.sphere[0:d] ) for d in D]
s_fam = [ fs.define_sphere_pen( dim = D[k], x_shift = x_shifts[k] ) for k in range(lD) ]
dim = 500
start = time.time()
t = 0
print("This computation may take a few minutes to complete, please be patient...")

for t in range(n_try):
    x_0s = fs.define_x_rand(count = n_pop, dim = dim, lim = [-80, 80])
    myResults = np.empty((25, 5), dtype = float)
    myXs = np.empty((25, dim), dtype=float)
    myResult = pso.minimize(s_fam[1], x_0s,
                            options = {'max_iter':max_iter,
                                       'verbose':True,
                                       'friction':friction,
                                       'max_velocity': max_velocity,
                                       'g_rate':g_rate,
                                       'l_rate':l_rate})
    myResults[t,0] = myResult.fun
    myResults[t,1] = myResult.nit
    myResults[t,2] = myResult.nsit
    myResults[t,3] = myResult.status
    myResults[t,4] = myResult.success
    myXs[t] = myResult['x']
end = time.time()

avgTime = (end - start) / t

print("fitness = ", myResults[:,0][0])
print('time to compute = ', avgTime)

input("Press enter to exit")