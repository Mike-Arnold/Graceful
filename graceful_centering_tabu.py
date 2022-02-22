import random as random
import time
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
#import os
import csv

def tree_to_matrix(tree,nodes):
    temp = []
    for line in nx.generate_adjlist(tree):
        #line is a string separated by spaces, turn it into a list
        my_list = line.split(" ")
        #first element is just a label, so remove it
        my_list.pop(0)
        #turn the list from strings to integers
        my_list = [int(i) for i in my_list]
        
        temp.append(my_list)
    #fill the matrix with zeroes
    matrix = [[0]*(nodes) for i in range(nodes)]
    
    #the ones indicate a connection
    #find where the ones should go
    for i in range(0,nodes):
        for j in range(0,nodes):
            if j in temp[i]:
                matrix[i][j] = 1
                matrix[j][i] = 1
    
    return matrix

def graph_tree(tree,id,nodes=None,seed=None,time=None,conflicts=0):
    plt.figure(id, figsize = (10,10))
    #options for making the graph look nicer
    pos = graphviz_layout(tree, prog="neato")
    #pos = graphviz_layout(tree, prog="dot")
    #pos = graphviz_layout(tree, prog="twopi")
    
    nx.draw(tree,pos,node_size=900)
    
    #set z = 2 if using even/odds, else set z=1
    z = 1
    
    #set node labels to be strictly positive, and perhaps all odd if setting z=2
    node_labels = {}
    for u in tree.nodes():
        key = u
        value = z*(u+1) - 1
        node_labels[key] = value
    
    #set edge labels to be the difference between the nodes it connects
    edge_labels = {}
    for u,v in tree.edges():
        key = (u,v)
        value = z*abs(u-v)
        edge_labels[key] = value
    
    font_size = 20
    nx.draw_networkx_labels(tree, pos, labels = node_labels, font_size = font_size)
    
    #the 'normal' way - looks bad
    #nx.draw_networkx_edge_labels(tree, pos, edge_labels = edge_labels, font_size = font_size)
    #this alternative keeps the edge labels from rotating
    text = nx.draw_networkx_edge_labels(tree, pos, edge_labels = edge_labels, font_size = font_size)
    for _,t in text.items():
        t.set_rotation('horizontal')
    
    if id == 1:
        title = str("Problem graph \n Nodes = {} \n Seed = {}".format(nodes,seed))
#        file_name = str("Problem_graph_{}_{}.png".format(nodes,seed))
    if id == 2:
        title = str("Local solution with value {} \n Nodes = {} \n Seed = {} \n Time to solve: {} seconds".format(conflicts,nodes,seed,time))
#        file_name = str("Solution_graph_{}_{}.png".format(nodes,seed))
    if id == 3:
        title = str("Global solution \n Nodes = {} \n Seed = {} \n Time to solve: {} seconds".format(nodes,seed,time))
#        file_name = str("Solution_graph_{}_{}.png".format(nodes,seed))
    plt.title(title,fontsize = font_size)
#    my_path = os.path.dirname(__file__)
    plt.gcf().set_facecolor("#A9A9A9")
#    plt.savefig(my_path + '\\graphs2\\' + file_name, bbox_inches='tight')
    plt.show()

def print_matrix(matrix):
    max_len = 1
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            this_num = matrix[i][j]
            if len(str(this_num)) > max_len:
                max_len = len(str(this_num))
    
    for i in range(len(matrix)):
        rowtext = ""
        for j in range(len(matrix[i])):
            this_num = matrix[i][j]
            space = max_len - len(str(this_num))
            rowtext = rowtext + space*' ' + str(this_num) + " "
        print(rowtext)

def assignWeights(matrix):
    #the weight of each tree will be based on the number of close neighbors
    #so, pick a node and find the distance to each other node
    #add up these distances - if it's a smaller number, that node is more central
    
    #note that the matrix already notes all the distance 1's
    #so use it as a starting point
    distances = [row[:] for row in matrix]
    
    for d in range(2,N):
        for i in range(N):
            distances[i][i] = -1
            for j in range(N):
                if distances[i][j] == d-1:
                    for k in range(N):
                        if matrix[j][k] == 1 and distances[i][k] == 0:
                            distances[i][k] = d
                            
    #the rowsums will be the weights for each node
    #smaller weighted nodes should be given more extreme node labels
    weights = []
    for i in range(N):
        rowsum = 0
        for j in range(N):
            if distances[i][j] == -1:
                distances[i][j] = 0
            rowsum += distances[i][j]
        weights.append(rowsum)
        
    return weights,distances
   
def judge(perm, matrix, weights, J):
    perm_matrix = perm_to_matrix(perm)
    new_matrix = reorder(perm_matrix, matrix)
    evaluation =  [f(element * judgeByWeights(perm,weights)) for element in conflicts(new_matrix)]
    # troublemakers = findTrouble(new_matrix)
    return evaluation #, troublemakers
    
def perm_to_matrix(perm):
    matrix = []
    for i in range(N):
        row = []
        for j in range(N):
            if perm[i] == j:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return matrix

def conflicts(tree):
    count = -N + 1
    for i in range(1,N): #start with the diagonal where C-R = 1
        d = 0
        for j in range(i,N):
            d += tree[j-i][j]
        count += d**2
        
    not_used = [1]*(N-1)
    for i in range(1,N): #start with the diagonal where C-R = 1
        for j in range(i,N):
            if tree[j-i][j] == 1:
                not_used[i-1] = 0

    unused_weighted = 0
    for i in range(1,N):
        unused_weighted += i * not_used[i-1]
        
    return int(count / 2), sum(not_used), unused_weighted
    
# def findTrouble(tree):
#     #identify where the 1s are
#     diagonals = []
#     rows = []
#     cols = []
#     for i in range(1,N):
#         for j in range(i,N):
#             if tree[j-i][j] == 1:
#                 diagonals.append(i)
#                 rows.append(j-i)
#                 cols.append(j)
    
#     troublemakers = []
#     #find which diagonals are used more than once
#     for i in range(N-1):
#         for j in range(i+1,N-1):
#             if diagonals[i] == diagonals[j]:
#                 troublemakers.append(rows[i]) #the troublemaker lived in this row
#                 troublemakers.append(rows[j])
#                 #troublemakers.append(cols[i]) #need to track the cols as well because
#                 #troublemakers.append(cols[j]) #  only the upper triangle was checked
                
#     return troublemakers

def judgeByWeights(this_list,weights):
    judgement = 0
    for i in range(N):
        extremity = abs( this_list[i] - (N-1)/2 )
        judgement += extremity * weights[i]
    return judgement / (N**2)

def reorder(order_matrix, tree):
    OM = mmult(order_matrix, tree)
    OMO = mmult(OM, T(order_matrix))
    return OMO
    
def mmult(a,b):
    product = []
    for i in range(N):
        product.append([])
        for j in range(N):
            prodsum = 0
            for k in range(N):
                prodsum += a[i][k] * b[k][j]
            product[i].append(prodsum)
    return product

def T(a):
  #transpose
  T = [[a[j][i] for j in range(N)] for i in range(N)]
  #credit for this solution goes to https://www.geeksforgeeks.org/transpose-matrix-single-line-python/
  return T

def matrix_to_AL(matrix, nodes):
    adjacency_list = []
    for i in range(nodes):
        temp = str(i)
        for j in range(nodes):
            foo = matrix[i][j]
            if(j>i and foo == 1):
                temp += " " + str(j)
        adjacency_list.append(temp)
    return adjacency_list

def f(number):
    return int(number*1000)/1000

def swap(perm, best, tabu, matrix, weights, N,J):
    population = []
    evaluations = []
    flips = []
    for i in range(2*N):
        new_perm = perm.copy()
        r1,r2 = random.sample(range(N), 2)
        new_perm[r2],new_perm[r1] = new_perm[r1],new_perm[r2]
        
        # evaluation, troublemakers = judge(new_perm, matrix, weights, J)
        evaluation = judge(new_perm, matrix, weights, J)
        evaluations.append(evaluation[J])
        flips.append([r1,r2])
        population.append([new_perm, evaluation])
        
    best_index = evaluations.index(min(evaluations))
    r1,r2 = flips[best_index]
    p = random.random()
    if [r1,r2] not in tabu or [r2,r1] not in tabu or p<.01 or min(evaluations) == 0:
        tabu.append([r1,r2])
        tabu.pop(0)
        return population[best_index]
    else:
        return perm, best

def findSolution(N,S,J):
    nodes = N
    maxIter = 1000
    
    seed = S
    tree = nx.random_tree(nodes,seed=seed)
    graph_tree(tree,1,nodes,seed)
    matrix = tree_to_matrix(tree,nodes)
    #print_matrix(matrix)
    weights, distances = assignWeights(matrix)
    #print(weights)
    
    perm = list(range(N))
    best = [(N**2) * 2 * sum(weights)] * 3  # initialize to a high value
    tabu = [[].copy()] * int(N/3)
    
    start = time.time()
    for i in range(maxIter):
        perm, best = swap(perm, best, tabu, matrix, weights, N,J)
        print(i,perm,best)
        if best[J] == 0:
            break
    end = time.time()
    
    solve_time = f(end - start)
    
    print("Best solution value:",best)
    if best[J] > 0:
        print("Local Optimal Solution Found!")
        id = 2
    else:
        print("Global Optimal Solution Found!")
        id = 3
    solution_matrix = reorder(perm_to_matrix(perm),matrix)
    print_matrix(solution_matrix)
    AL = matrix_to_AL(solution_matrix, nodes)
    solution_tree = nx.parse_adjlist(AL, nodetype=int)
    graph_tree(solution_tree,id,nodes,seed,solve_time,best[J])
    
    # revert solutions to the same metric
    returnable_solution = [f(element / judgeByWeights(perm,weights)) for element in best]
    
    return solve_time, id, returnable_solution, i

if __name__ == '__main__':
    nMin = 100
    nMax = 100
    nInc = 5
    
    sMin = 27
    sMax = 28
    seeds = sMax-sMin
    seedList = [0,2,5,7,10,11,13,14,15,16,21,23,26,27,29,30,31,33,36,41,42,43,48,51,57,63,66,69,74,75,77,78,80,83,85,89,91,92,96,98]
    
    J = judgement_choice = 2 # 0=conflicts , 1=unused , 2=unused_weighted
    
    times = []
    solves = []
    
    conflict_sum = 0
    unused = 0
    unused_weighted = 0
    
    # with open('results_centering_tabu.csv', mode='w', newline='') as results_file:
    #     results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     results_writer.writerow(["Perfect","Time","Nodes","Seed","Evaluator","Conflicts","Unused","Unused_Weighted"])
    
    N = nMin
    while N <= nMax:
        totalTime = 0
        perfectSolves = 0
        imax = 0
        for S in seedList[sMin:sMax]:
            solveTime, id, judgements, i = findSolution(N,S,J)
            
            imax = max(imax,i)
            
            totalTime += solveTime
            perfectSolves += (id-2)
            
            conflict_sum += judgements[0]
            unused += judgements[1]
            unused_weighted += judgements[2]
            
            # with open('results_centering_tabu.csv', mode='a+', newline='') as results_file:
            #     results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     results_writer.writerow([id-2,solveTime,N,seedList.index(S),J,judgements[0],judgements[1],judgements[2]])
    
        print(imax)
    
        average = f(totalTime/seeds)
        times.append(average)
        solves.append(perfectSolves)
        N += nInc
    
    print()
    
    N = nMin
    i=0
    while N <= nMax:
        print("Centering")
        print("Judgement choice:",J)
        print("Average time for",N,"nodes:",times[i],"seconds")
        print("\t Perfect solutions:",solves[i],"out of",seeds)
        print("\t Average conflicts remaining:",f(conflict_sum/seeds))
        print("\t Average unused edge labels:",f(unused/seeds))
        print("\t Average unused (weighted):",f(unused_weighted/seeds))
        N += nInc
        i += 1