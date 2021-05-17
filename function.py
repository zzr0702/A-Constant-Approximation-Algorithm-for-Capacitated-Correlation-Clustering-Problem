import numpy as np
from pulp import *
from sklearn.datasets import load_iris

def G_random(N):
    N = N
    random = np.random.random((N,N))
    for i in range(N):
        for j in range(i+1):
            if i==j:
                random[i,j] = 0
            elif random[i,j] >= 0.5:
                random[i,j] = 1
            else:
                random[i,j] = -1
            random[j,i] = random[i,j]
    G = np.tril(np.array(random,dtype=int))
    return G

def G_iris(N):
    data = load_iris().target
    sampleNo = N
    num = np.random.choice(np.arange(len(data)),sampleNo,replace=False)
    iris_data = data[num]
    iris_mat = np.zeros((sampleNo,sampleNo))

    for i in range(sampleNo):
        for j in range(i+1,sampleNo):
            if iris_data[i] == iris_data[j]:
                iris_mat[i][j] = 1
            else:
                iris_mat[i][j] = -1

    iris_mat = iris_mat.T
    return iris_mat

def G_US(N):
    sampleNo = N
    data_raw = np.loadtxt('data1000.txt',delimiter=',').astype(int)
    data = data_raw[:,1]
    num = np.random.choice(data,sampleNo,replace=False)
    US_data = data[num]
    US_mat = np.zeros((sampleNo,sampleNo))
    for i in range(sampleNo):
        for j in range(i+1,sampleNo):
            if US_data[i] == US_data[j]:
                US_mat[i][j] = 1
            else:
                US_mat[i][j] = -1
    return US_mat.T

def G_heart(N):
    sampleNo = N
    data_raw = np.loadtxt('Heart.txt',delimiter=',').astype(int)
    data = data_raw[:,-1]
    num = np.random.choice(data,sampleNo,replace=False)
    H_data = data[num]
    H_mat = np.zeros((sampleNo,sampleNo))
    for i in range(sampleNo):
        for j in range(i+1,sampleNo):
            if H_data[i] == H_data[j]:
                H_mat[i][j] = 1
            else:
                H_mat[i][j] = -1
    return H_mat.T

def solutionLP(graph,capacity):
    """

    :param G: randomly generate undirected graph satisfying the constrains
    :param U: the capacity of each group
    :return: optimal solution and optimal value
    """
    G = graph
    N = np.shape(G)[0]
    U = capacity
    prob = LpProblem('Problem',sense=LpMinimize)
    var = [[LpVariable(f'x{i}{j}',lowBound=0,upBound=1,cat=LpContinuous ) for j in range(i+1)] for i in range(N)]
    flatten = lambda x : (y for l in x for y in flatten(l)) if type(x) is list else [x]
    coefficient = np.array([G[i,j] for i in range(N) for j in range(i+1)])

    # objective function
    prob += lpDot(flatten(var),coefficient) + len(np.where(coefficient==-1)[0])

    # constraint 1 : triangle inequality
    combination_point = [(x,y,z)for x in range(N) for y in range(x+1,N) for z in range(y+1,N)]
    for item in combination_point:
        var1 = var[item[1]][item[0]]
        var2 = var[item[2]][item[0]]
        var3 = var[item[2]][item[1]]
        prob += var1 + var2 >= var3
        prob += var1 + var3 >= var2
        prob += var2 + var3 >= var1

    # constraint 2 : capacity
    for i in range(N):
        constrains = []
        for j in range(N):
            if i >= j:
                constrains.append(1-var[i][j])
            else:
                constrains.append(1-var[j][i])
        prob += lpSum(constrains) <= U

    # constrains 3 : zeros
    for i in range(N):
        prob += var[i][i] == 0

    prob.solve()

    # data storage
    opt_solution = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1):
            opt_solution[i,j] = value(var[i][j])
    # np.savetxt('opt_solution.csv',opt_solution,delimiter=',',fmt='%.2f')
    opt_value = value(prob.objective)
    return opt_solution,opt_value

def solutionIP(graph,capacity):
    """

    :param G: randomly generate undirected graph satisfying the constrains
    :param U: the capacity of each group
    :return: optimal solution and optimal value
    """
    G = graph
    U = capacity
    N = np.shape(G)[0]
    prob = LpProblem('Problem',sense=LpMinimize)
    var = [[LpVariable(f'x{i}{j}',cat=LpBinary ) for j in range(i+1)] for i in range(N)]
    flatten = lambda x : (y for l in x for y in flatten(l)) if type(x) is list else [x]
    coefficient = np.array([G[i,j] for i in range(N) for j in range(i+1)])

    # objective function
    prob += lpDot(flatten(var),coefficient) + len(np.where(coefficient==-1)[0])

    # constraint 1 : triangle inequality
    combination_point = [(x,y,z)for x in range(N) for y in range(x+1,N) for z in range(y+1,N)]
    for item in combination_point:
        var1 = var[item[1]][item[0]]
        var2 = var[item[2]][item[0]]
        var3 = var[item[2]][item[1]]
        prob += var1 + var2 >= var3
        prob += var1 + var3 >= var2
        prob += var2 + var3 >= var1

    # constraint 2 : capacity
    for i in range(N):
        constrains = []
        for j in range(N):
            if i >= j:
                constrains.append(1-var[i][j])
            else:
                constrains.append(1-var[j][i])
        prob += lpSum(constrains) <= U

    # constrains 3 : zeros
    for i in range(N):
        prob += var[i][i] == 0

    prob.solve()

    # data storage
    opt_solution = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1):
            opt_solution[i,j] = value(var[i][j])
    opt_value = value(prob.objective)
    return opt_solution,opt_value

def algorithm(graph,capacity,solution):
    alpha = (-5 + np.sqrt(33))/2
    G = graph
    U = capacity
    N = np.shape(G)[0]
    V = {}
    opt_solution = solution
    opt_solution = opt_solution + opt_solution.T
    S = list(np.arange(N))
    while len(S):
        v_random = np.random.choice(S)
        v_random_vector = opt_solution[v_random]
        sum = 0
        index_list = []
        index_delete = []
        v_vector_sortIndex = np.argsort(v_random_vector)
        v_vector_sortValue = v_random_vector[v_vector_sortIndex]
        for item,i in zip(v_vector_sortValue,v_vector_sortIndex):
            if item <= alpha and i in S and len(index_list)< U:
                sum += item
                index_list.append(opt_solution[i])
                index_delete.append(i)
        V['v_{}'.format(len(V)+1)] = {}
        if sum/len(index_list) >= alpha/2 :
            V['v_{}'.format(len(V))][v_random] = v_random_vector
            S.remove(v_random)
        else:
            for item,i in zip(index_list,index_delete):
                V['v_{}'.format(len(V))][i] = item
                S.remove(i)

    cluster_point = [[x for x in V[y]] for y in V]
    matching = [(x,y) for x in range(len(V)) for y in range(x+1,len(V))]
    alg_solution = np.zeros((N,N))
    for item in matching:
        for x in cluster_point[item[0]]:
            for y in cluster_point[item[1]]:
                alg_solution[x][y] = 1
                alg_solution[y][x] = 1
    alg_solution = np.tril(alg_solution)
    alg_value = np.sum(np.multiply(alg_solution,G)) + len(np.where(G==-1)[0])
    return alg_solution,alg_value




