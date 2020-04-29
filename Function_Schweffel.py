#***********************************************************************************
#******************************SCHWEFFEL's FUNCTION*************************************
#***********************************************************************************
print("Schweffel")

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
n_try = 5

# generate random shifts, and initial values of x and functions
x_shifts = [fs.define_x_rand(count = n_try, dim = d) for d in D]
x_0s = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_family_schwef(x_shifts[k]) for k in range(len(x_shifts))]

#***********************************************************************************
#******************************GRADIENT DESCENT*************************************
#***********************************************************************************


print("\n Estimated gradient descent with random data in dimension 50 :")

# gradient with unknown function
# with random data

x_opts = [np.empty((n_try, D[i]), dtype=float) for i in range(lD)]
f_opts = np.empty((lD, n_try), dtype=float)
n_runs = np.empty((lD, n_try), dtype=float)

# dimension 50
eps = 1 
ro = 7
max_run = 1000
i = 0

start = time.time()
dim = D[i]

print("\n sequence of 5 trials")

for t in range(n_try):
    print(dim, t)
    x_0 = x_0s[i][t]
    f = s_fam[i][t]
    (x_opts[i][t], f_opts[i][t], n_runs[i][t]) \
        = fs.gradient_method(dim, x_0, f, eps=eps, ro=ro, 
                                termination_criterion = fs.termination_by_runs, 
                                max_run = max_run, eps_adjust=True, ro_adjust=True, 
                                ro_optim=False)

end = time.time()
avg = (end - start) / n_try
print('average duration of 1 dim 50 = ', avg)

print("fitnesses = ", f_opts[0])
print("number of iterations = ", n_runs[0])

# dimension 500

print("\n Estimated gradient descent with random data in dimension 500 :")

eps = 1
ro = 50
max_run = 3000
i = 1

start = time.time()
dim = D[i]

print("\n sequence of 5 trials")

for t in range(n_try):
    print(dim, t)
    x_0 = x_0s[i][t]
    f = s_fam[i][t]
    (x_opts[i][t], f_opts[i][t], n_runs[i][t]) \
        = fs.gradient_method(dim, x_0, f, eps=eps, ro=ro, 
                                termination_criterion = fs.termination_by_runs, 
                                max_run = max_run, ro_adjust=True, ro_optim=False, 
                                eps_adjust = True)

end = time.time()
avg = (end - start) / n_try
print('average duration of 1 dim 500 = ', avg)

print("fitnesses = ", f_opts[1])
print("number of iterations = ", n_runs[1])


# with given data

print("\n Estimated gradient descent with real data in dimension 50 and 500 :")

n_try = 1
x_shifts = [ np.array(data.schwefel[0:d]) for d in D]
x_0s = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_schwef(dim = D[k], x_shift = x_shifts[k]) for k in range(lD)]
x_opts = [np.empty((n_try, D[i]), dtype=float) for i in range(lD)]
f_opts = np.empty((lD, n_try), dtype=float)
n_runs = np.empty((lD, n_try), dtype=float)

ro_ = [7, 50]
max_run_ = [2000, 3000]
eps_ajust = True


start = time.time()
for i in range(lD):
    dim = D[i]
    f = s_fam[i]
    ro = ro_[i]
    max_run = max_run_[i]
    for t in range(n_try):
        print(dim, t)
        x_0 = x_0s[i][t]
        (x_opts[i][t], f_opts[i][t], n_runs[i][t]) \
            = fs.gradient_method(dim, x_0, f, eps=eps, ro=ro, 
                                 termination_criterion = fs.termination_by_runs, 
                                 max_run=max_run, eps_adjust=eps_ajust, ro_adjust=True, ro_optim=False)

end = time.time()
avg = (end - start) / n_try
print('average duration of 1 dim 50 + 1 dim 500 = ', avg)

print("fitnesses = ", f_opts)
print("number of iterations = ", n_runs)
print( "errors : " )
print(np.linalg.norm(x_opts[0][0] - x_shifts[0]))
print(np.linalg.norm(x_opts[1][0] - x_shifts[1]))

#***********************************************************************************
#******************************PSO*************************************************
#***********************************************************************************

# PSO with random data

# set constraints
const = [{'type':'ineq', 'fun': lambda x: x[i] - L[0]} for i in range(Dmax) ]
const.extend( [ {'type':'ineq', 'fun': lambda x: -x[i] + L[1]} for i in range(Dmax) ] )

# generate random shifts, and initial values of x and functions
n_pop = 50
n_try = 200

x_shifts = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_family_schwef(x_shifts[k]) for k in range(lD)]

# test learning rates in dimension 50
rmin = 0.5
rmax = 5.5
ncheck = 11
max_iter = 200
friction = 0.5
step = (rmax - rmin) / (ncheck - 1)
rates = np.empty((ncheck, ncheck))
values = np.arange(rmin, rmax+0.1, step)

g = 0

print("\n With PSO :")
print("\n Try different learning rates in dimension 50 on random shifts")

for grate in values:
    l = 0
    for lrate in values:
        x_0s = [fs.define_x_rand(count = n_pop, dim = d) for d in D]
        i = (ncheck * l + g) % 25
        f = s_fam[0][i]
        myResult = pso.minimize(f, x_0s[0], 
                                options = {'max_iter':max_iter,
                                          'verbose':False,
                                          'friction':friction,
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

# implement with data

print("\n PSO with real data in dimension 50, 25 trials with different initial points")

n_try = 25

n_pop = 50
max_iter = 1500
g_rate = 4
l_rate = 4
friction = 0.9
max_velocity = 2
x_shifts = [ np.array( data.schwefel[0:d] ) for d in D]
s_fam = [ fs.define_schwef_pen( dim = D[k], x_shift = x_shifts[k] ) for k in range(lD) ]
callback = lambda x: fs.fit_in_bounds( x = x, bounds = L)
myResults = np.empty((25, 5), dtype = float)

dim = 50
start = time.time()
for t in range(n_try):
    print(dim, t)
    x_0s = fs.define_x_rand(count = n_pop, dim = dim)
    myXs = np.empty((25, dim), dtype=float)
    myResult = pso.minimize(s_fam[0], x_0s,
                            options = {'max_iter':max_iter,
                                       'verbose':False,
                                       'friction':friction,
                                       'max_velocity': max_velocity,
                                       'g_rate':g_rate,
                                       'l_rate':l_rate},
                            callback=callback)
    myResults[t,0] = myResult.fun
    myResults[t,1] = myResult.nit
    myResults[t,2] = myResult.nsit
    myResults[t,3] = myResult.status
    myResults[t,4] = myResult.success
    myXs[t] = myResult.x
end = time.time()

# see results
avgTime = (end - start) / t
print('average time for 1 trial =', avgTime)

print("\n Fitnesses :")
print(myResults[:,0])

x_init = np.average(myXs, axis = 0)

print("\n Perform gradient descent on the barycentre of all 25 final points")

xl, fl, nl = fs.gradient_method(dim, x_init, s_fam[0], eps=10**-3, ro=7, 
                                 termination_criterion = fs.termination_by_runs, 
                                 max_run=1000, ro_adjust=True, ro_optim=False)

print("Final fitness = ", fl, " ; after ", nl, " runs")
print("Error = ", np.linalg.norm(xl-x_shifts[0]))

# scaling in higher dimension 500

# test learning rates

n_pop = 100
n_try = 200

x_shifts = [fs.define_x_rand(count = n_try, dim = d) for d in D]
s_fam = [fs.define_family_schwef(x_shifts[k]) for k in range(lD)]

rmin = 0.5
rmax = 5.5
ncheck = 11
step = (rmax - rmin) / (ncheck - 1)
rates = np.empty((ncheck, ncheck))
values = np.arange(rmin, rmax+0.1, step)
callback = lambda x: fs.fit_in_bounds( x = x, bounds = L)
max_iter = 300
friction = 0.5

print("\n Try different learning rates in dimension 500 on random shifts")

g = 0
for grate in values:
    l = 0
    for lrate in values:
        x_0s = [fs.define_x_rand(count = n_pop, dim = d) for d in D]
        i = (ncheck * l + g) % 25
        f = s_fam[1][i]
        myResult = pso.minimize(f, x_0s[1], 
                                options = {'max_iter':max_iter,
                                          'verbose':False,
                                          'friction':friction,
                                          'g_rate':grate,
                                          'l_rate':lrate},
                                callback=callback)
        rates[l, g] = myResult.fun
        print("local rate = ", lrate,
             " ; global rate = ", grate,
            " ; fitness = ", rates[l, g])
        l += 1
    g += 1

rates2 = rates
plt.imshow(rates2, cmap=cm.gray)
plt.title(label = "Fitness of PSO according to learning rate \n \
            (global_rate*2, local_rate*2) \
            \n in dimension 500")
plt.show()


# with real data

print("\n PSO with real data in dimension 500")

n_pop = 100
n_try = 100
max_iter = 100
g_rate = 4
l_rate = 4
friction = 0.9
max_velocity = 2

# implement constraints through a call back
callback = lambda x: fs.fit_in_bounds(x = x, bounds = L)

# real data
# we penalize the objective function to avoid checking constraints
x_shifts = [ np.array( data.schwefel[0:d] ) for d in D]
s_fam = [ fs.define_schwef_pen( dim = D[k], x_shift = x_shifts[k] ) for k in range(lD) ]
dim = 500

print("\n Implement PSO with real data in dimension 500, 100 short trials")
print(" (results to be used to initialize next step)")


myResults = np.empty((n_try, 5), dtype = float)
start = time.time()
t = 0
for t in range(n_try):
    print(dim, t)
    x_0s = fs.define_x_rand(count = n_pop, dim = dim, lim = [-100, 100])
    myXs = np.empty((n_try, dim), dtype=float)
    myResult = pso.minimize(s_fam[1], x_0s,
                            options = {'max_iter':max_iter,
                                       'verbose':False,
                                       'friction':friction,
                                       'max_velocity': max_velocity,
                                       'g_rate':g_rate,
                                       'l_rate':l_rate},
                            callback=callback)
    myResults[t,0] = myResult.fun
    myResults[t,1] = myResult.nit
    myResults[t,2] = myResult.nsit
    myResults[t,3] = myResult.status
    myResults[t,4] = myResult.success
    myXs[t] = myResult['x']
end = time.time()

avgTime = (end - start)
print('time to compute =', avgTime)

print("\n Implement PSO in dimension 500 starting with the final points of previous runs")

max_iter = 1000
g_rate = 4
l_rate = 0.2
max_velocity = 0.5
friction = 0.5
myResult2 = pso.minimize(s_fam[1], myXs,
                         options = {'max_iter':max_iter,
                                    'verbose':False,
                                    'friction':friction,
                                    'max_velocity': max_velocity,
                                    'g_rate':g_rate,
                                    'l_rate':l_rate})

print("Final fitness = ", myResult2.fun)
print("Error = ", np.linalg.norm(myResult2.x - x_shifts[1]))

input("Press enter to quit")