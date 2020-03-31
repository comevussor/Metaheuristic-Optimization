# load tools
import os
import random
import numpy as np
from scipy.spatial import distance_matrix as distM
import math
import abc

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution
import jmetal.algorithm.singleobjective as so
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.util.ckecking import Check
import copy
from typing import List, TypeVar
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import BinaryTournamentSelection

# TSP problem definition (based on jmetalpy improved to import distance matrix directly)
class TSP2(PermutationProblem):
    """ Class representing TSP Problem personnalized. """

    def __init__(self, coord_matrix):
        super(TSP2, self).__init__()
        self.coord_matrix = coord_matrix
        self.distance_matrix = self._compute_distance()
        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = len(coord_matrix)
        self.number_of_objectives = 1
        self.number_of_constraints = 0

    def _compute_distance(self):
        # Euclidian distance matrix
        computed_matrix = distM(self.coord_matrix, self.coord_matrix)
        return computed_matrix


    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitness = 0

        for i in range(self.number_of_variables - 1):
            x = solution.variables[i]
            y = solution.variables[i + 1]

            fitness += self.distance_matrix[x][y]

        first_city, last_city = solution.variables[0], solution.variables[-1]
        fitness += self.distance_matrix[first_city][last_city]

        solution.objectives[0] = fitness

        return solution

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives)
        new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)

        return new_solution

    @property
    def number_of_cities(self):
        return self.number_of_variables

    @property
    def matrix_of_distances(self):
        return self.distance_matrix

    def get_name(self):
        return 'Symmetric TSP'

# Mutation definition (contains an index error in jmetalpy)
class PermutationSwapMutation(Mutation[PermutationSolution]):

    def __init__(self, probability: float, randMut, D=0, n=0, first=True):
        super(PermutationSwapMutation, self).__init__(probability=probability)
        self.randMut = randMut
        self.D = D
        self.n = n
        self.first = first

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        Check.that(type(solution) is PermutationSolution, "Solution type invalid")
        rand = random.random()
        
        if rand <= self.probability:

            if self.randMut == 1:
                # pos_one, pos_two = random.sample(range(solution.number_of_variables - 1), 2)
                # there is no use for the -1 above in the original algorithm
                pos_one, pos_two = random.sample(range(solution.number_of_variables), 2)

            elif self.randMut == 2:
                path = solution.variables
                pos_one, pos_two, length  = bestMutation2(self.D, self.n, path, self.first)

            solution.variables[pos_one], solution.variables[pos_two] = solution.variables[pos_two], solution.variables[pos_one]

        return solution
           
    def get_name(self):
        return 'Permutation Swap mutation'

    

class PMXCrossover(Crossover[PermutationSolution, PermutationSolution]):

    def __init__(self, probability: float):
        super(PMXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        permutation_length = offspring[0].number_of_variables

        rand = random.random()
        if rand <= self.probability:
            cross_points = sorted([random.randint(0, permutation_length) for _ in range(2)])

            def _repeated(element, collection):
                c = 0
                for e in collection:
                    if e == element:
                        c += 1
                return c > 1

            def _swap(data_a, data_b, cross_points):
                c1, c2 = cross_points
                new_a = data_a[:c1] + data_b[c1:c2] + data_a[c2:]
                new_b = data_b[:c1] + data_a[c1:c2] + data_b[c2:]
                return new_a, new_b

            def _map(swapped, cross_points):
                n = len(swapped[0])
                c1, c2 = cross_points
                s1, s2 = swapped
                map_ = s1[c1:c2], s2[c1:c2]
                for i_chromosome in range(n):
                    if not c1 < i_chromosome < c2:
                        for i_son in range(2):
                            while _repeated(swapped[i_son][i_chromosome], swapped[i_son]):
                                map_index = map_[i_son].index(swapped[i_son][i_chromosome])
                                swapped[i_son][i_chromosome] = map_[1 - i_son][map_index]
                return s1, s2

            swapped = _swap(parents[0].variables, parents[1].variables, cross_points)
            mapped = _map(swapped, cross_points)

            offspring[0].variables, offspring[1].variables = mapped

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Partially Matched crossover'

    
S = TypeVar('S')

class RouletteWheelSelection(Selection[List[S], S]):
    """Performs roulette wheel selection.
    """

    def __init__(self, beta):
        super(RouletteWheelSelection).__init__()
        self.beta = beta

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        if self.beta == 0:
            # maximum = sum([solution.objectives[0] for solution in front]) # defaulf formula on jmetalpy
            maximum = sum([1/solution.objectives[0] for solution in front])
        else:
            maximum = sum([math.exp(- self.beta * solution.objectives[0]) for solution in front])

        rand = random.uniform(0.0, maximum)
        value = 0.0

        for solution in front:
            if self.beta == 0:
                # value += solution.objectives[0] # defaul formula in jmetalpy
                value += 1/solution.objectives[0]
            else:
                value += math.exp(-self.beta * solution.objectives[0])

            if value > rand:
                return solution

        return None

    def get_name(self) -> str:
        return 'Roulette wheel selection'



def tryTSP(problem, population_size, offspring_population_size, mutation, crossover, selection, termination_criterion):
    myAlgo = so.GeneticAlgorithm(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion)

    myAlgo.run()
    result = myAlgo.get_result()

    print('Algorithm: {}'.format(myAlgo.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    print('Solution: {}'.format(result.variables))
    print('Fitness: {}'.format(result.objectives[0]))
    print('Computing time: {}'.format(myAlgo.total_computing_time))

    return result

def findClosest(D, M, from_ind, among_ind):
    dist = M + 1
    neighb = from_ind

    for i in among_ind :
        if D[from_ind, i] < dist :
            dist = D[from_ind, i]
            neighb = i

    return (neighb, dist)

def findOptPath(D, n, M, from_city = 0):
    city = from_city
    totDist = 0
    arrDist = np.zeros(n)
    availableCities = np.arange(n, dtype=int)
    path = np.zeros(n, dtype=int)
    path[0]=from_city

    for i in range(1, n):
        availableCities = availableCities[availableCities != city]
        (neighb, dist) = findClosest(D, M, city, availableCities)
        path[i] = neighb
        arrDist[i-1] = dist
        totDist += dist
        city = neighb

    arrDist[i] = D[from_city, city]
    totDist += D[from_city, city]

    return (totDist, path, arrDist)

def findGLobalOptPath(D, n, optimize = False, maxLoop=100, verbal = False, first = False):
    vecDist = np.zeros(n)
    matOptPath = np.zeros((n,n), dtype=int)
    matArrDist = np.zeros((n,n))
    M = np.amax(D, axis = None)

    for from_city in range(n):
        (vecDist[from_city], matOptPath[from_city,:], matArrDist[from_city,:]) = \
            findOptPath(D, n, M, from_city)

        if optimize:
            (matOptPath[from_city,:], vecDist[from_city]) = \
                optimizePath(D, n, matOptPath[from_city,:], verbal=verbal, maxLoop=maxLoop, first=first)

    return(vecDist, matOptPath, matArrDist)


# optimize best neighbour with best mutation (this method is too costly)
def bestMutation(D, n, path, first=False):

    refLen = pathLen(path, D, n)
    myMin = refLen
    test = myMin
    mut1, mut2 = 0, 0
    proba = 10/n

    for i in range(n-1):
        for j in range(i+1,n):
            tmp = copy.deepcopy(path)
            tmp[i] = path[j]
            tmp[j] = path[i]
            tmpInd = np.array([i-1, i, i+1, j-1, j, j+1], dtype=int)%n

            test = refLen - D[path[tmpInd[0]], path[tmpInd[1]]] - D[path[tmpInd[1]], path[tmpInd[2]]] \
                - D[path[tmpInd[3]], path[tmpInd[4]]] - D[path[tmpInd[4]], path[tmpInd[5]]] \
                + D[tmp[tmpInd[0]], tmp[tmpInd[1]]] + D[tmp[tmpInd[1]], tmp[tmpInd[2]]] \
                + D[tmp[tmpInd[3]], tmp[tmpInd[4]]] + D[tmp[tmpInd[4]], tmp[tmpInd[5]]] 

            if  test <= myMin :
                rand = (random.random() < proba )
                if rand:
                    (mut1, mut2, myMin) = (i, j, test)
                    if first:
                        return (mut1, mut2, myMin)

    return (mut1, mut2, myMin)

# optimize best neighbour with best mutation (this method is optimized with numpy arrays)
def bestMutation2(D, n, path, first=False):    
    refLen = pathLen(path, D, n)
    myMin = refLen
    mut1, mut2 = 0, 0
    Lpath = np.array(path).tolist() # convert in cases of getting a path as an array
    Dpath = D[Lpath, :][:, Lpath]
    
    d0 = np.diagonal(Dpath, 1)
    d_loop = Dpath[0, n-1]
    d1 = np.append( d0, d_loop)

    slideP = np.roll(np.arange(n), 1).tolist()
    slideM = np.roll(np.arange(n), -1).tolist()

    # for non consecutive mutations (i, i+3 at least)
    loss0 = d1 + d1[slideP]
    loss1 = np.array([loss0,]*n) + np.array([loss0,]*n).transpose()

    gain1 = Dpath[slideM, :] + Dpath[slideP, :] + Dpath[:, slideM] + Dpath[:, slideP]

    mutationMat1 = gain1 - loss1

    # for consecutive mutations (i, i+1)
    loss01 = d1[slideM] + d1[slideP]
    loss2 = np.diag(loss01[0:(n-1)], 1)
    loss2[0,n-1] = loss01[n-1]

    gain21 = Dpath[slideP, :] + Dpath[:, slideM]
    gain2 = np.diag( np.diag( gain21, 1 ), 1 )
    gain2[0, n-1] = gain21[0, n-1]

    mutationMat2 = gain2 - loss2
    
    mutationMat = np.triu(mutationMat1, 2) + np.diag( np.diag( mutationMat2, 1 ), 1 )
    mutationMat[0, n-1] = Dpath[0, n-2] + Dpath[1, n-1] - Dpath[0, 1] - Dpath[n-2, n-1]

    improvedMut = mutationMat[mutationMat<0]
    improvedShape = improvedMut.shape

    if improvedShape != (0,):
        if (first == False):
            bestPair = np.argwhere(mutationMat == np.min(mutationMat))[0]
            (mut1, mut2, myMin) = (bestPair[0], bestPair[1], mutationMat[bestPair] + refLen)

        elif (first == True):
            choose = np.random.choice(improvedMut)
            bestPair = np.argwhere(mutationMat == choose)[0]

    return (mut1, mut2, myMin)

def pathLen(path, D, n):
    dist = D[path[0], path[n-1]]

    for i in range(n-1):
        dist += D[path[i], path[i+1]]

    return(dist)

def mutate(path, i, j):
    path[i], path[j] = path[j], path[i]

    return path

def optimizePath(D, n, path, verbal = True, maxLoop = 100, first=False ):
    go_on = True
    currentPath = copy.deepcopy(path)
    count = 0

    while go_on & (count < maxLoop) :
        (mut1, mut2, myMin) = bestMutation2(D, n, path=currentPath,first=first)

        if verbal:
            print(myMin)

        if mut1 != mut2 :
            currentPath = mutate(currentPath, mut1, mut2)

        count += 1

    print(myMin)

    return (currentPath, myMin)
