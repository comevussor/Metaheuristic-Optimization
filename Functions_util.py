import numpy as np
from numpy.random import default_rng
rng = default_rng(1611) # insert seed here 1611


# randomly get a sample of x_shifts
def define_x_rand(count = 1, dim = 2, lim = [-100, 100] ):
    return rng.random( ( count, dim ) ) * (lim[1] - lim[0]) + lim[0]

# clip a vector in a domain
def fit_in_bounds(x, bounds):
    x = np.clip(x, bounds[0], bounds[1])
    return x

#****************************************************************************************
#*******************************GENERATE SHIFTED SPHERE FUNCTIONS************************
#****************************************************************************************

# Utilities about sphere function
def define_sphere(dim, x_shift, y_shift = -450):

    def func(x) -> float:
        return sum( ( x - x_shift) ** 2 ) + y_shift

    return func

# penalized sphere function
def define_sphere_pen(dim, x_shift, y_shift = -450, lim = [-100, 100]):

    def func(x) -> float:
        pen = sum( np.maximum(x-lim[1], 0) + np.maximum(lim[0]-x, 0) ) ** 2
        return sum( ( x - x_shift) ** 2 ) + y_shift + pen

    return func

# note that x_shift must be a family of vectors
def define_family_sphere(x_shifts):    

    sh = x_shifts.shape
    s_fam = [ define_sphere( dim = sh[1], x_shift = x_shifts[i] ) for i in range( sh[0] ) ]

    return s_fam

#****************************************************************************************
#*************************************GENERATE SCHWEFEL FUNCTION***************************
#****************************************************************************************

def define_schwef(dim, x_shift, y_shift = -450):

    def func(x) -> float:
        return np.max( np.abs( x - x_shift)  ) + y_shift

    return func

# penalized Schwefel function
def define_schwef_pen(dim, x_shift, y_shift = -450, lim = [-100, 100]):

    def func(x) -> float:
        pen = (10 * sum( np.maximum(x-lim[1], 0) + np.maximum(lim[0]-x, 0) )) ** 3
        return np.max( np.abs( x - x_shift)  ) + y_shift + pen

    return func

# note that x_shift must be a family of vectors
def define_family_schwef(x_shifts):    

    sh = x_shifts.shape
    s_fam = [ define_schwef( dim = sh[1], x_shift = x_shifts[i] ) for i in range( sh[0] ) ]

    return s_fam

#****************************************************************************************
#*****************************GENERATE ROSENBROCK FUNCTION************************************
#****************************************************************************************

def define_rosen(dim, x_shift, y_shift = -390):

    def func(x) -> float:
        xx = x - x_shift + 1
        dd = dim - 1
        return np.sum( 100 * ( xx[:dd]**2 - xx[1:] ) ** 2 + ( xx[:dd] - 1 ) ** 2 ) + y_shift

    return func

# penalized Rosenbrock function
def define_rosen_pen(dim, x_shift, y_shift = -390, lim = [-100, 100]):

    def func(x) -> float:
        xx = x - x_shift + 1
        dd = dim - 1
        pen = (10 * sum( np.maximum(x-lim[1], 0) + np.maximum(lim[0]-x, 0) )) ** 3
        ros = np.sum( 100 * ( xx[:dd]**2 - xx[1:] ) **2 + ( xx[:dd] - 1 ) ** 2 ) + y_shift
        return ros + pen

    return func

# note that x_shift must be a family of vectors
def define_family_rosen(x_shifts):    

    sh = x_shifts.shape
    s_fam = [ define_rosen( dim = sh[1], x_shift = x_shifts[i] ) for i in range( sh[0] ) ]

    return s_fam

#****************************************************************************************
#*******************************GENERATE RASTRIGIN'S FUNCTION*****************************
#****************************************************************************************

# define Rastrigin function
def define_rast(dim, x_shift, y_shift = -330):

    def func(x) -> float:
        xx = x - x_shift
        return np.sum( xx ** 2 - 10 * np.cos(2 * np.pi * xx )  + 10 ) + y_shift

    return func

# define Rastrigin function to accept array of variables
# x_shift is a vector
def define_rast_multi(dim, x_shift, y_shift = -330):

    def func(X) -> float:
        XX = X - x_shift
        return np.sum( XX ** 2 - 10 * np.cos(2 * np.pi * XX )  + 10 , axis=1) + y_shift

    return func

# penalized Rastrigin function
def define_rast_pen(dim, x_shift, y_shift = -330, lim = [-5, 5]):

    def func(x) -> float:
        xx = x - x_shift
        pen = (10 * sum( np.maximum( x - lim[1], 0) + np.maximum( lim[0] - x, 0) ) ) ** 3
        rast = np.sum( xx ** 2 - 10 * np.cos(2 * np.pi * xx )  + 10 ) + y_shift
        return rast + pen

    return func

# note that x_shift must be a family of vectors
def define_family_rast(x_shifts):    

    sh = x_shifts.shape
    s_fam = [ define_rast( dim = sh[1], x_shift = x_shifts[i] ) for i in range( sh[0] ) ]

    return s_fam

#****************************************************************************************
#*****************************ESTIMATED GRADIENT DESCENT METHOD*************************
#****************************************************************************************
# estimate gradient
def estim_grad(dim, x, f, eps):

    Meps = np.diag( eps * np.ones( dim ) ) / 2
    x_1 = x - Meps
    x_2 = x + Meps
    gradient = np.zeros(dim)

    for i in range(dim):
        gradient[i] = ( f( x_2[i] ) - f( x_1[i] ) ) / eps

    return gradient

def termination_by_runs(k_run, x_i, x_j, max_run = 0, min_step = 1):
    return  ( k_run >= max_run )

def termination_by_step(k_run, x_i, x_j, max_run = 0, min_step = 1):
    return ( np.linalg.norm(x_j - x_i) < min_step )


# implement gradient descent
def gradient_method(dim, x_0, f, eps, ro, termination_criterion, 
                    max_run = 0, min_step = 1, bounds = [-100, 100],
                   ro_adjust = False, ro_optim = False, eps_adjust = False, 
                   verbal=False):

    hard_max_run = 10000
    max_run = min( [ max_run, hard_max_run ] )
    x_cur = x_0
    g = estim_grad(dim, x_cur, f, eps)
    x_opt = x_0 - ro * g
    x_opt = fit_in_bounds(x_opt, bounds)
    n_run = 1
    ro_cur = ro

    while n_run < hard_max_run \
        and ( not termination_criterion(n_run, x_cur, x_opt, max_run=max_run, min_step=min_step) ):

        x_cur = x_opt
        ro_cur = ro
        g = estim_grad(dim, x_cur, f, eps)
        norm_g = np.linalg.norm(g)

        if norm_g > 0:
            g = g / norm_g

        if ro_adjust:
            if n_run == (np.floor (max_run / 2)):
                ro = ro / 10
                ro_cur = ro_cur / 10

            if n_run == (np.floor(max_run / 4)):
                ro = ro / 10
                ro_cur = ro_cur / 10

        if eps_adjust:
            if n_run == (np.floor (max_run / 2)):
                eps = eps / 10

            if n_run == (np.floor(max_run / 4)):
                eps = eps / 10

        if ro_optim:
            ro_cur = ro_optimize(dim, x_cur, f, g, ro_cur) * ro_cur

        x_opt = x_cur - ro_cur * g
        x_opt = fit_in_bounds(x_opt, bounds)
        n_run += 1

        if verbal:
            print(x_opt, f(x_opt), n_run)

    return (x_opt, f(x_opt), n_run)

# implement optimal look for optimal step
def ro_optimize(dim, x_0, f, g, ro, max_iter = 1000):
    x_opt = np.copy(x_0)
    f_opt = f(x_0)
    x_cur = x_opt - ro * g 
    f_cur = f(x_cur)
    n = 0

    while f_cur < f_opt and n < max_iter:
        n += 1
        x_opt, f_opt = x_cur, f_cur
        x_cur = x_cur - ro * g
        f_cur = f(x_cur)

    return np.max([n, 1])


#****************************************************************************************
#**********************************SIMULATED ANNEALING***********************
#****************************************************************************************

def simulateAnnealing2(value, initState, maxIter=1000, divers=1, intens=10**-3, 
                       accept=1, bounds = [-100, 100], verbal=False):

# value : fitness (energy) function, takes x only and returns a real number
# maxIter : maximum number of iterations
# initState : initial positions of particles
# divers : diversifying factor to increase perturbations
# intens : intensifying factor to target minimal perturbation
# accept : acceptance factor, to be set greater for lower acceptance

# perturbations are scaled from divers to intens along the max_run iterations

    # initialize 
    dim = len(initState) # dimension of the problem
    iter = 0 # iterations count

    curState = initState # current state
    curVal = value(curState) # current energy

    minState = curState # store state of minimum energy found
    minVal = curVal # state minimum energy found
    minIter = 0 # iteration count when minimum was found

    verb = 0 # indicator of verbality

    # as long as temperature and iterations count allow
    while (iter < maxIter):
        luck = False

        # perturbate current state more if temperature is high
        perturb = perturbation(dim)
        scaling = divers * intens ** ( iter / maxIter )

        # scaling = divers * t
        nextState = curState + scaling * perturb
        nextState = fit_in_bounds(nextState, bounds)
        nextVal = value(nextState) 

        # compute variation of energy
        nrjVar = nextVal - curVal

        # if energy is improved or on random basis for reasonable energy variation,
        #  accept new state and cool down
        improve = ( nrjVar < 0 )
        if not improve:
            # luck = ( ( np.exp(- np.min( [ nrjVar, 10 ** 5 ] )) / 2 / np.log10( iter + 1 ) ) >= np.random.random() )
            luck = ( ( np.exp(- np.min( [ nrjVar, 10 ** 5 ] ) * accept ) ) >=  \
                np.random.random() )
        
        if improve or luck:
            curState = nextState
            curVal = nextVal

        if nextVal < minVal :
            minVal, minState = nextVal, nextState
            minIter = iter

            if iter > (verb + 100) and (verbal == 1) :
                verb = iter
                print(iter, nextVal)

        # increase iterations count
        iter += 1

    result = {"iter":minIter,
              "last_state":curState,
              "last_energy":curVal,
              "minimum_state":minState,
              "minimum_energy":minVal
              }
    if verbal == 2  :
        print("Minimum energy = ", minVal)

    return(result)


# get a perturbation vector of length n with values between -1 and 1
def perturbation(dim):
    p = 2 * np.random.random((dim)) - 1
    return p

#****************************************************************************************
#********************************SIMPLEX METHOD*************************
#****************************************************************************************

# initial simplex is right-angled at X with step h
def initSimplex( x, dim, step ):
    return np.concatenate( ( [x], step * np.identity(dim) + [x] * dim), axis=0 )

# compute image of a simplex without ordering
def simplexImage( simplex, fun ):
    return np.array( [ fun(x) for x in simplex ] )

# order acording to Nelder-Mead : max, 2nd max and min
def orderSimplex( simplex, values, dim ):

    indices = np.argsort(values)
    return { "value_max": values[ indices[ dim ] ],
            "x_max": simplex[ indices[ dim ] ],
            "index_max": indices[ dim ],
            "value_2nd_max": values[ indices[ dim - 1 ] ],
            "x_2nd_max": simplex[ indices[ dim - 1 ] ],
            "index_2nd_max": indices[ dim - 1 ],
            "value_min": values[ indices[0] ],
            "x_min": simplex[ indices[0] ],
            "index_min": indices[0]
            }

# get isobarycenter of simplex minus worst point
def getCentroid( simplex, index_max, dim ):
    return sum( np.delete( simplex, index_max, axis=0 ) ) / dim
    # return sum( np.delete( simplex, np.random.randint(0, dim+1), axis=0 ) ) / dim

# TRANSFORMATIONS
def reflect( centroid, x_max, c_reflect=-1, bounds=[-5, 5] ):
    image = centroid + c_reflect * ( x_max - centroid )
    image = fit_in_bounds(image, bounds)
    return image

def expand( centroid, x_reflected, c_expand=1.5, bounds=[-5, 5] ):
    image = centroid + c_expand * ( x_reflected - centroid ) 
    image = fit_in_bounds(image, bounds)
    return image

def contract( centroid, x_reflected, c_contract=0.5):
    return centroid + c_contract * ( x_reflected - centroid ) 

# shrink towards best point after transformations
def shrink( dim, simplex, index_min, c_shrink=0.5 ):
    x_min = simplex[ index_min ]
    xx_min = [ x_min ] * ( dim + 1 )
    return xx_min + c_shrink * ( simplex - xx_min )

def initialize( dim, fun, initMethod=0, initGuess=0, step=1, bounds=[-5, 5]):

    if (initMethod == 0):
        guess = np.random.rand(dim) * ( bounds[1] - bounds[0] ) + bounds[0]
        simplex = initSimplex(guess, dim, step)

    elif initMethod == 1:
        simplex = initSimplex(initGuess, dim, step)

    elif initMethod == 2:
        simplex = initGuess

    values = simplexImage(simplex, fun)
    order = orderSimplex( simplex, values, dim )

    return {"simplex":simplex, "value":values, "order":order} 

# replace worst point with new
def replace(simplex, values, index, x, x_value):

    simplex2 = np.copy( simplex )
    value2 = np.copy( values )

    simplex2[ index ] = x
    value2[ index ] = x_value

    return simplex2, value2

# transform the simplex once
def transformSimplex( dim, simplex, values, order, range, fun, c_reflect, 
                     c_expand, c_contract, c_shrink, bounds):

    mustShrink = False
    centroid = getCentroid( simplex, order[ "index_max" ], dim )
    x_reflected = reflect( centroid, order[ "x_max" ], c_reflect, bounds )
    value_reflected = fun( x_reflected )


    # if value_reflected is between the 2nd max and the min, just keep it
    if ( order[ "value_min" ] <= value_reflected ) and ( value_reflected < order[ "value_2nd_max" ] ):
        x_new = [ x_reflected, value_reflected ]

    # if reflection improves the minimum, push it a bit further to try to improve more
    elif ( value_reflected < order[ "value_min" ] ):
        x_expanded = expand( centroid, x_reflected, c_expand, bounds )
        value_expanded = fun( x_expanded )

        # if expansion improves reflection, keep expension, otherwise keep reflection
        if value_expanded < value_reflected:
            x_new = [ x_expanded, value_expanded ]
        else:
            x_new = [ x_reflected, value_reflected ]

    # from here value_reflected >= value_2nd_max
    # if value_reflected is between max and 2nd max, contract reflection towards centroid
    elif value_reflected < order[ "value_max" ]:
        x_contracted = contract( centroid, x_reflected, c_contract )
        value_contracted = fun( x_contracted )

        # if contraction improves, keep contraction, otherwise, nothing to do, shrink the simplex
        if value_contracted <= value_reflected:
            x_new = [ x_contracted, value_contracted ]
        else:
            mustShrink = True

    # otherwise, it means that value_reflected is greater than max, contract x_max rather than reflected
    else:
        x_contracted = contract( centroid, order[ "x_max" ], c_contract )
        value_contracted = fun( x_contracted )

        # if contraction improves value, keep it, otherwise, nothing to do, shrink the simplex
        if value_contracted <= value_reflected:
            x_new = [ x_contracted, value_contracted ]
        else:
            mustShrink = True

    if mustShrink:
        simplex2 = shrink( dim, simplex, order[ "index_min" ], c_shrink )
        value2 = simplexImage( simplex2, fun )

    else:
        simplex2, value2 = replace(simplex, values=values, index=order[ "index_max" ], x=x_new[0], x_value=x_new[1] )

    order2 = orderSimplex( simplex2, value2, dim )

    return {"simplex":simplex2, "value":value2, "order":order2}

# implement the algorithm iteratively
def NelderMead_simplex (dim, initial_state, max_iter, min_range, fun,
                        c_reflect=-1, c_expand=1.5, c_contract=0.5, c_shrink=0.5, bounds=[-5, 5], verbal=False):

    iter = 0
    range = np.abs( initial_state[ "order" ][ "value_max" ] - initial_state[ "order" ][ "value_min" ] )
    current_state = initial_state

    while ( iter < max_iter ) and ( range > min_range ):

        current_state = transformSimplex(dim, current_state[ "simplex" ],
                                        current_state[ "value" ],
                                        current_state[ "order" ], range,
                                        fun, c_reflect, c_expand, c_contract, c_shrink, bounds)
        
        range = np.abs( current_state[ "order" ][ "value_max" ] - current_state[ "order" ][ "value_min" ] )
        iter += 1

        if verbal and ( iter%50 == 0 ):
            print(iter, current_state[ "order" ][ "value_min" ], range )


    result = {"iterations": iter,
              "range": range,
              "fitness": current_state[ "order" ][ "value_min" ],
              "end_point": current_state[ "order" ][ "x_min" ],
              "simplex":current_state[ "simplex" ]}

    print(result)
    return result