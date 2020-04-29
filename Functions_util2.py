# Load useful libraries

import numpy as np
import numpy.random as nprd
# from numpy.random import default_rng
rng = nprd.default_rng(1611) # insert seed here 1611 for reproducibility

#*********************************************************************************************
#************************************General purpose functions********************************

# Get a sample of randomly generated shifts in an hypercube
def define_x_rand(count, dim, lim):
    # count (integer) : giving sample size
    # dim (integer) : dimension
    # lim (list of 2 floats) : boundaries typically [-M, M]
    return rng.random( ( count, dim ) ) * (lim[1] - lim[0]) + lim[0]


#******************Define shifted Rastrigin function according to shifts************************
def define_rast_multi(x_shift, y_shift):
    # x_shift (1-D array of floats)
    # y_shift (float)

    def func(X, axis = 0) -> float:
        # X (array of floats)
        XX = X - x_shift
        return np.sum( XX ** 2 - 10 * np.cos(2 * np.pi * XX )  + 10 , axis) + y_shift

    return func

#******************Define shifted Griewank's function according to shifts*************************
def define_griewank_multi(x_shift, y_shift):
    # x_shift (1-D array of floats)
    # y_shift (float)

    def func(X, axis) -> float:
        # X (1-D or 2-D array of floats)
        XX = X - x_shift 
        XX = np.reshape( XX, (-1, np.shape(X)[-1] ) )
        sh = np.shape(XX)
        sq = np.reshape( 1 / np.sqrt( np.arange( 1, sh[ 1 ] + 1 ) ), ( 1, -1 ) )

        on = np.ones( ( sh[0], 1 ) )
        coef = np.matmul( on, sq  )
        cos_arg = np.multiply( coef, XX )
        cos_mat = np.cos( cos_arg )
        minus_mat = np.prod( cos_mat , axis=1 )
        norm_mat = np.sum( XX ** 2, axis=1 )
        result = 1 / 4000 * norm_mat - minus_mat + 1 + y_shift

        return np.reshape(result, np.shape(X)[0:-1] )

    return func

#***********************************************************************************************
#*******************************BATCH GRADIENT DESCENT*****************************************
#***********************************************************************************************

# Estimate gradient value
def estim_grad(dim, X, f, eps):
    # dim (integer) : dimension
    # X (2-D array of float)
    # f (X, axis = 0) : X (array of floats), axis for Rastrigin sum, returns array of floats
    # eps (small float) : step to evaluate

    n = len(X)
    Xplus = np.array( [ np.identity(dim) *  eps / 2 + x  for x in X ] )
    Xminus = np.array( [ - np.identity(dim) *  eps / 2 + x  for x in X ] )

    grad = ( f(Xplus, axis = 2) - f(Xminus, axis = 2) ) / eps
    return grad

# Check termination conditions
def check_termination(n_run, max_run, step, min_step, terminate_by):
    if terminate_by == 'runs':
        return n_run < max_run
    elif terminate_by == 'step':
        return np.all( step > min_step )
    else:
        return False

# IMPLEMENT GRADIENT DESCENT
def gradient_descent(dim, lim, X_0, f, eps, ro, terminate_by = 'runs', 
                     max_run = 10, min_step = 1., max_step = 10., verbal = False):
    # dim (integer) : dimension
    # lim (list of 2 floats) : domain boundaries
    # X_0 (2-D array of reals) : set of initial points
    # f : function to optimize takes any array of reals and axis 
    #       to identify antecedents
    # eps (float) : x step to compute approximated gradient
    # ro (float) : move = - ro * gradient
    # terminate_by ('runs' or 'step')
    # max_run (integer)
    # min_step, max_step (floats)
    # verbal (boolean)

    n = len(X_0)
    X_cur = X_0
    grad = estim_grad(dim, X_cur, f, eps)
    move = ro * grad

    # step = length of move (2-norm)
    step = np.linalg.norm( move, axis=1 )

    # ajdust move according to max_step
    move = np.array( [ move[i] * max_step / np.max ( [ max_step, step[i] ]) 
                      for i in range(n) ] )

    # clip new x in hypercube
    X_next = np.clip( X_cur + move, lim[0], lim[1] )

    # initialize runs count
    n_run = 1

    while check_termination(n_run, max_run, step, min_step, terminate_by):
        X_cur = X_next
        grad = estim_grad(dim, X_cur, f, eps)
        move = ro * grad
        step = np.linalg.norm( move, axis=1 )
        move = np.array( [ move[i] * max_step / np.max ( [ max_step, step[i] ]) 
                          for i in range(n) ] )
        X_next = np.clip( X_cur - move, lim[0], lim[1] )
        n_run += 1

        if verbal:
            print(X_next, f(X_next, axis = 1), n_run) 

    return (X_next, f(X_next, axis = 1), n_run)

#simplified version for parallelization

def para_gradient(dim, lim, f, eps, ro, max_run, max_step, verbal):
    def func(X_0):
        return gradient_descent(dim, lim, X_0, f, eps, ro, max_run=max_run, 
                                max_step=max_step, verbal=verbal)
    return func

#*********************************************************************************************
#******************  Simulated annealing for a set of points  ********************************
#*********************************************************************************************

def simulated_annealing(dim, lim, X_0, f, max_run=10000, divers=1, 
                        intens=10**-3, accept=1, verbal=False):
    # dim (integer) : dimension
    # lim (list of 2 floats) : domain boundaries
    # X_0 (2-D array of reals) : set of initial points
    # f : function to optimize takes any array of reals and axis 
    #       to identify antecedents
    # divers : diversifying factor to increase perturbations
    # intens : intensifying factor to target minimal perturbation
    # accept : acceptance factor, to be set greater for lower acceptance
    # max_run (integer)
    # verbal (boolean)

    # define a sie for batches of random numbers generation
    batch = 1000

    # perturbations are scaled from divers to intens along the max_run iterations

    n = len(X_0)
    n_ones = np.ones(n) # to broadcast a value n times

    X_cur = X_0
    Y_cur = f( X_0, axis=1 )
    
    # X_min records the best x for each path (starting from each initial point)
    X_min = X_cur
    # Y_min records the corresponding minimum values of the function
    Y_min = Y_cur
    # run_min records the number of runs after which the minimum value has been found
    run_min = np.zeros(n)

    # initialize runs count
    n_run = 0

    # generate 1000 perturbations between -1 and 1
    # and 1000 random luck values between 0 and 1
    batch_perturb = np.random.random( (batch, n, dim) ) * 2 - 1
    batch_luck = np.random.random( ( batch, n ) )
    batch_iter = 0    

    for n_run in range(max_run):

        perturb = batch_perturb[ batch_iter ]
        luck = batch_luck[ batch_iter ]

        # compute scaling factor
        scaling = divers * intens ** ( n_run / max_run )

        # update points and clip
        X_next = np.clip( X_cur + scaling * perturb, lim[0], lim[1] )
        Y_next = f( X_next, axis=1 )

        # record global improvements
        improve = ( Y_next < Y_min ) *1

        # update minima
        X_min = np.array( [ ( 1 - improve[i]) * X_min[i] + improve[i] * X_next[i] 
                           for i in range(n) ])
        Y_min = (1 - improve) * Y_min + improve * Y_next
        run_min = (1 - improve) * run_min + improve * n_run * n_ones

        # function variation (clip it for exponential limitations)
        Y_var = np.clip( Y_next - Y_cur, -10**4, 10**4)

        choice = np.array( [ 1 if Y_var[i] < 0 else ( luck[i] <= \
            np.exp(- accept * Y_var[i] ) ) * 1 
                            for i in range(n) ] )

        # reject old ones and update
        X_cur = np.array( [ X_next[i] * choice[i] + X_cur[i] * (1-choice[i]) 
                           for i in range(n) ] )
        Y_cur = np.array( [ Y_next[i] * choice[i] + Y_cur[i] * (1-choice[i]) 
                           for i in range(n) ] )

        batch_iter += 1
        if batch_iter == batch:
            batch_perturb = np.random.random( (batch, n, dim) ) * 2 - 1
            batch_luck = np.random.random( ( batch, n ) )
            batch_iter = 0

        if verbal and ( ( n_run % 500 ) == 0 ) :
            print("n_run = ", n_run)
            print("Y_var = ", Y_var)
            print("choice = " , choice)
            print("Y_min", Y_min)
            print("")

    return [X_min, Y_min, run_min]
    
# simplified function for parallelization
def para_anneal (dim, lim, f, max_run, divers, intens, accept, verbal):

    def func(X_0):
        return simulated_annealing(dim, lim, X_0, f, max_run=max_run, divers=divers, 
                                   intens=intens, accept=accept, verbal=verbal)

    return func

#*********************************************************************************************
#**************************************SIMPLEX METHOD*****************************************
#*********************************************************************************************

# initial simplex is right-angled at X with step h
def initSimplex( x, dim, step ):
    return np.concatenate( ( [x], step * np.identity(dim) + [x] * dim), axis=0 )

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

# shrink towards best point after transformations
def shrink( dim, simplex, index_min, c_shrink, bounds):
    x_min = simplex[ index_min ]
    xx_min = [ x_min ] * ( dim + 1 )
    image =  xx_min + c_shrink * ( simplex - xx_min )
    image = np.clip(image, bounds[0], bounds[1])
    return image

def initialize( dim, fun, initMethod=0, initGuess=0, step=1, bounds=[-5, 5]):

    if (initMethod == 0):
        guess = np.random.rand(dim) * ( bounds[1] - bounds[0] ) + bounds[0]
        simplex = initSimplex(guess, dim, step)

    elif initMethod == 1:
        simplex = initSimplex(initGuess, dim, step)

    elif initMethod == 2:
        simplex = initGuess

    values = fun( simplex, axis = 1)
    order = orderSimplex( simplex, values, dim )

    return {"simplex":simplex, "values":values, "order":order} 

def transformSimplex( dim, simplex, values, order, fun, 
                     c_reflect, c_expand, c_contract, c_shrink, bounds, verbal ):
    mustShrink = False

    # get isobarycenter of simplex minus worst point
    centroid = np. sum( np.delete( simplex, order[ "index_max"] , axis=0 ), axis=0) / dim

    # reflect worst with respect to centroid
    x_reflected = np.clip( centroid + c_reflect * ( order["x_max"] - centroid ), 
                          bounds[0], bounds[1])
    value_reflected = fun( x_reflected, axis = 0)


    # if value_reflected is between the 2nd max and the min, just keep it
    if ( order[ "value_min" ] <= value_reflected ) and \
        ( value_reflected < order[ "value_2nd_max" ] ):
        x_new = [ x_reflected, value_reflected ]

    # if reflection improves the minimum, push it a bit further to try to improve more
    elif ( value_reflected < order[ "value_min" ] ):
        x_expanded = np.clip(centroid + c_expand * ( x_reflected - centroid ), 
                             bounds[0], bounds[1])
        value_expanded = fun( x_expanded )

        # if expansion improves reflection, keep expension, otherwise keep reflection
        if value_expanded < value_reflected:
            x_new = [ x_expanded, value_expanded ]
        else:
            x_new = [ x_reflected, value_reflected ]

    # from here value_reflected >= value_2nd_max
    # if value_reflected is between max and 2nd max, contract reflection towards centroid
    elif value_reflected < order[ "value_max" ]:
        x_contracted = np.clip(centroid + c_contract * ( x_reflected - centroid ) , 
                               bounds[0], bounds[1])
        value_contracted = fun( x_contracted )

        # if contraction improves, keep contraction, otherwise, nothing to do, shrink the simplex
        if value_contracted <= value_reflected:
            x_new = [ x_contracted, value_contracted ]
        else:
            mustShrink = True

    # otherwise, it means that value_reflected is greater than max, contract x_max rather than reflected
    else:
        x_contracted = np.clip(centroid + c_contract * ( order[ "x_max" ] - centroid ) , 
                               bounds[0], bounds[1])
        value_contracted = fun( x_contracted )

        # if contraction improves value, keep it, otherwise, nothing to do, shrink the simplex
        if value_contracted <= value_reflected:
            x_new = [ x_contracted, value_contracted ]
        else:
            mustShrink = True

    if mustShrink:
        simplex = shrink( dim, simplex, order[ "index_min" ], c_shrink, bounds)
        values = fun( simplex, axis=1 )

    else:
        # replace worst point with new
        simplex[ order [ "index_max" ] ] = x_new[0]
        values[ order [ "index_max" ] ] = x_new [1]

    order = orderSimplex( simplex, values, dim )

    return {"simplex":simplex, "values":values, "order":order}

def NelderMead_simplex (dim, initial_state, max_iter, min_range, fun,
                        c_reflect=-1, c_expand=1.5, c_contract=0.5, 
                        c_shrink=0.5, bounds=[-5, 5], verbal=False):

    iter = 0
    range = np.abs( initial_state[ "order" ][ "value_max" ] - \
        initial_state[ "order" ][ "value_min" ] )
    current_state = initial_state

    while ( iter < max_iter ) and ( range > min_range ):

        current_state = transformSimplex(dim, current_state[ "simplex" ],
                                        current_state[ "values" ],
                                        current_state[ "order" ],
                                        fun, c_reflect, c_expand, c_contract, 
                                        c_shrink, bounds, verbal)
        
        range = np.abs( current_state[ "order" ][ "value_max" ] - \
            current_state[ "order" ][ "value_min" ] )
        iter += 1

        if verbal and ( iter%100 == 0 ):
            print(iter, current_state[ "order" ][ "value_min" ], range )


    result = {"iterations": iter,
              "range": range,
              "fitness": current_state[ "order" ][ "value_min" ],
              "end_point": current_state[ "order" ][ "x_min" ],
              "simplex":current_state[ "simplex" ]}

    print(result)
    return result

#*********************************************************************************************
# ****************************BARYCENTRIC SIMULATED ANNEALING*********************************
#*********************************************************************************************
# MUST BE with len(X_0)=dim+1 otherwise moving in a subspace

def simulated_annealing_bary(dim, lim, X_0, f, max_run=10000, accept=1, neg_margin=0.1, 
                             neighbours=10, min_main_weight=0.25, order=False, 
                             verbal=False):
    # dim (integer) : dimension
    # lim (list of 2 floats) : domain boundaries
    # X_0 (2-D array of reals) : set of initial points
    # f : function to optimize takes any array of reals and axis 
    #       to identify antecedents
    # divers : diversifying factor to increase perturbations
    # intens : intensifying factor to target minimal perturbation
    # accept : acceptance factor, to be set greater for lower acceptance
    # max_run (integer)
    # verbal (boolean)

    # define a size for batches of random numbers generation
    batch = 1000

    # perturbations are scaled from divers to intens along the max_run iterations

    n = len(X_0)

    X_cur = np.copy( X_0 )
    X_next = np.copy( X_0 )
    Y_cur = f( X_0, axis=1 )
    
    # X_min records the best x for each path (starting from each initial point)
    X_min = np.copy(X_cur)
    # Y_min records the corresponding minimum values of the function
    Y_min = np.copy(Y_cur)
    # run_min records the number of runs after which the minimum value has been found
    run_min = np.zeros(n)

    # initialize runs count
    n_run = 0

    # generate batch random luck values between 0 and 1
    batch_luck = np.random.random( ( batch, n ) )
    batch_iter = 0    

    for n_run in range(max_run):

        # update points and clip
        weights = get_weights(n, neighbours, min_main_weight, neg_margin )
        neighbours_list = get_neighbours(n, neighbours)

        if order:
            for i in range(n):
                neighbours_fit = Y_cur[ neighbours_list[i, 1:] ]
                indexes = np.argsort( neighbours_fit ) + 1
                neighbours_list[i, 1:] = neighbours_list[i, indexes ]
                weights[i, 1:] = - np.sort( - weights[i, 1:] )

        # print(neighbours_list[0:5])
        # print(weights[0:5])
        # print([Y_cur[neighbours_list[i,:] ] for i in range(5)] )
        
        for i in range(n):
            X_next[i] = np.clip( np.matmul( [ weights[i] ], X_cur[ neighbours_list[i] ] ), 
                                lim[0], lim[1] )

        Y_next = f( X_next, axis=1 )
        
        # function global variation (clip it for exponential limitations)
        Y_var_global = np.clip( Y_next - Y_min, -10**4, 10**4)

        # update global minima
        yes_index_global = np.argwhere( Y_var_global < 0 )
        X_min [ yes_index_global ] = X_next [ yes_index_global ]
        Y_min [ yes_index_global ] =  Y_next [ yes_index_global ]
        run_min [ yes_index_global ] = n_run

        # function local variation (clip it for exponential limitations)
        Y_var = np.clip( Y_next - Y_cur, -10**4, 10**4)

        # record local improvements
        improve = ( Y_var < 0 )
        not_index = np.argwhere( Y_var >= 0)
        n_not_improve = np.sum( Y_var >= 0 )
        luck = batch_luck[ batch_iter ]

        for i in range( n_not_improve ):
            improve [ not_index[i] ]  = ( luck[i] <= \
                np.exp(- accept * Y_var[ not_index[i] ] ) )

        # reject old ones and update
        yes_index = np.argwhere( improve ==  True )
        X_cur [ yes_index ] = X_next [ yes_index ]
        Y_cur [ yes_index ] = Y_next [ yes_index ]

        batch_iter += 1
        # generate batch random luck values when necessary
        if batch_iter == batch:
            batch_luck = np.random.random( ( batch, n ) )
            batch_iter = 0

        if verbal:
            print("n_run = ", n_run)
            n_yes_improve = np.sum(improve)
            print("sum choice = " , n_yes_improve )
            print(min(Y_min))

    return [X_min, Y_min, run_min]

# BARYCENTRIC PERTURBATION
def get_weights(pop_size, neighbours, min_main_weight = 0.25, neg_margin = 0):
    # weights sum to 1 but range between -neg_margin and 1
    # weights decrease exponentially from the considered point's weight
    # we use the main point and a set of randomly chosen 9 points from the set

    tot_weights = 1 + 2 * neg_margin

    weights = np.random.random( ( pop_size, neighbours + 1 ) ) 
    weights[:, 0] = min_main_weight
    weights[:, 1] *=  tot_weights - weights[:, 0]

    for i in range( 2, neighbours + 1 ):
        weights[:, i] *= (tot_weights - np.sum( weights[:, 0:i ], axis=1 ))

    if neg_margin > 0 :

        for j in range(pop_size):
            cum_w = weights[j, 0]
            
            i = 0
            while (cum_w < 1 + neg_margin) and (i < neighbours):
                i += 1
                cum_w += weights[j, i]

            if i < neighbours:
                weights[ j, ( i + 1 ):neighbours ] *= - 1
                weights[j, i] = 1 - np.sum( np.delete( weights[j], i ) )
            else:
                weights[j, neighbours] = 1 - np.sum( weights[j, 0:neighbours] )

    else:
        weights[:, neighbours] = 1 - np.sum( weights[:, 0:neighbours ], axis=1)

    return weights

def get_neighbours(pop_size, neighbours):
    # get neighbours random integers between 1 and pop_size-1

    neighbours_list = np.random.randint(1, pop_size - 1, (pop_size, neighbours))

    # 0 does not appear
    # shift by the index and modulo pop_size => index does not appear
    broadcast = np.transpose( [ range( pop_size) ] )
    neighbours_list += broadcast
    neighbours_list = np.mod(neighbours_list,  pop_size )
    neighbours_list  = np.append( broadcast, neighbours_list,  axis=1)

    return neighbours_list

# # simplified function for parallelization
def para_anneal_bary (dim, lim, f, max_run, accept, neg_margin, 
                      neighbours, min_main_weight, verbal, order):

    def func(X_0):
        return simulated_annealing_bary(dim, lim, X_0, f, max_run = max_run, accept=accept,
                                        neg_margin=neg_margin, neighbours=neighbours,
                                        min_main_weight=min_main_weight, verbal=verbal)

    return func

#***********************************************************************************************
#*******************************NESTED GRADIENT DESCENT*****************************************
#***********************************************************************************************


def nested_gradient(dim, lim, f, x_0, eps_0, ro_0, terminate_by_0='runs', max_run_0=100, 
                    min_step_0=1., max_step_0=10, verbal_0=False, eps=0.01, ro=0.05, 
                    terminate_by='runs', max_run=20, min_step=0, max_step=0.2, 
                    verbal=False, record = False) :
    
    def grad_fit(X_0):

        result = gradient_descent(dim = dim, lim = lim, X_0 = X_0, f = f, eps = eps, 
                                  ro = ro, terminate_by = terminate_by, max_run = max_run,
                                  min_step = min_step, max_step = max_step, verbal = verbal)
        return result
    
    # initialize local variables
    x_cur = np.copy(x_0)
    x_min = np.copy(x_cur)
    y_min = f(x_min, axis=0)
    move_mat = np.append( - eps_0 * np.identity(dim), eps_0 * np.identity(dim), axis=0 )
    n_run = 0
    n_run_min = 0
    step = min_step_0 + 1

    while check_termination(n_run, max_run_0, step, min_step_0, terminate_by_0) :

        X_multi = np.tile( x_cur, ( 2 * dim, 1 ) )
        X_move = move_mat + X_multi
        X_mat = np.append([x_cur], X_move, axis=0 )
        XX_mat, Y_list, info_run = grad_fit(X_mat)
        
        x_cur = XX_mat[0]
        y_cur = Y_list[0]

        if record and ( y_cur < y_min ) :
            x_min = x_cur
            y_min = y_cur
            n_run_min = n_run

        estim_grad = ( Y_list[ ( 1 + dim ): ] - Y_list[ 1:( 1 + dim ) ] ) / ( 2 * eps_0 )
        estim_grad = estim_grad / np.linalg.norm( estim_grad )

        move = ro_0 * estim_grad
        step = np.linalg.norm(move)

        x_cur -= move
        n_run += 1

        if verbal_0 :
            print ("n_run = ", n_run)
            print( "x_cur = ", x_cur)
            print( "y_cur = ", f(x_cur, axis=0))

    if record :
        return( x_min, y_min, n_run_min)
    else :
        return( x_cur, f(x_cur, axis=1), n_run)