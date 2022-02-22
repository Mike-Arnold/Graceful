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
    for i in range(nodes):
        for j in range(nodes):
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
    #    file_name = str("Problem_graph_{}_{}.png".format(nodes,seed))
    if id == 2:
        title = str("Local solution with value {} \n Nodes = {} \n Seed = {} \n Time to solve: {} seconds".format(conflicts,nodes,seed,time))
    #    file_name = str("Solution_graph_{}_{}.png".format(nodes,seed))
    if id == 3:
        title = str("Global solution \n Nodes = {} \n Seed = {} \n Time to solve: {} seconds".format(nodes,seed,time))
    #    file_name = str("Solution_graph_{}_{}.png".format(nodes,seed))
    plt.title(title,fontsize = font_size)
    #my_path = os.path.dirname(__file__)
    plt.gcf().set_facecolor("#A9A9A9")
    #plt.savefig(my_path + '\\graphs2\\' + file_name, bbox_inches='tight')
    plt.show()

### Matrix manipulation

def reorder(order_matrix, tree):
    OM = mmult(T(order_matrix), tree)
    OMO = mmult(OM, order_matrix)
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
        
#######################

def assignWeights(problem_matrix):
    #the weight of each tree will be based on the number of close neighbors
    #so, pick a node and find the distance to each other node
    #add up these distances - if it's a smaller number, that node is more central
    
    #note that the problem_matrix already notes all the distance 1's
    #so use it as a starting point
    distances = [row[:] for row in problem_matrix]
    for d in range(2,N):
        for i in range(N):
            distances[i][i] = -1
            for j in range(N):
                if distances[i][j] == d-1:
                    for k in range(N):
                        if problem_matrix[j][k] == 1 and distances[i][k] == 0:
                            distances[i][k] = d
    
    weights = []
    for i in range(N):
        rowsum = 0
        for j in range(N):
            if distances[i][j] == -1:
                distances[i][j] = 0
            rowsum += distances[i][j]
        weights.append(rowsum)
        
    return weights,distances
        
def judge(this_list, tree, weights, distances):
    this_matrix = listToMatrix(this_list)
    newTree = reorder(this_matrix, tree)
    evaluation = conflicts(newTree)
    troublemakers = findTrouble(newTree)
    return evaluation, troublemakers
    
def listToMatrix(this_list):
    this_matrix = []
    for i in range(N):
        this_row = []
        for j in range(N):
            if this_list[i] == j:
                this_row.append(1)
            else:
                this_row.append(0)
        this_matrix.append(this_row)
    return this_matrix

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
    
def findTrouble(tree):
    #identify where the 1s are
    diagonals = []
    rows = []
    cols = []
    for i in range(1,N):
        for j in range(i,N):
            if tree[j-i][j] == 1:
                diagonals.append(i)
                rows.append(j-i)
                cols.append(j)
    
    troublemakers = []
    for i in range(N-1):
        for j in range(i+1,N-1):
            if diagonals[i] == diagonals[j]:
                troublemakers.append(rows[i]) #the troublemaker lived in this row
                troublemakers.append(rows[j]) #the other one lived in this row
                #troublemakers.append(cols[i]) #need to track the cols as well because
                #troublemakers.append(cols[j]) #  only the upper triangle was checked
                
    return troublemakers

def chooseConnection(weights, distances):
    # first version will always choose the best-looking candidate
    # but this one may not always produce a solution
    # so future versions should have a chance to choose the second or third best
    
    # appraisals = [x[:] for x in distances]  # make a deep copy
    # min_value = -1
    # node_a = -1
    # node_b = -1
    
    # for a in range(N):
    #     for b in range(N):
    #         if appraisals[a][b] == 1:
    #             appraisals[a][b] = weights[a] + weights[b]  #plus or multiply?
    #             if min_value == -1 or appraisals[a][b] < min_value:
    #                 min_value = appraisals[a][b]
    #                 node_a,node_b = a,b
    #         else:
    #             appraisals[a][b] = 0
    
    # and this is the future version
    # a problem: using distance is very good at finding the 'best-looking' splitting edge
    # but it will often choose useless leafs as the second-best-looking
    
    appraisals = []
    appraisal_indexes = []
    for a in range(N-1):
        for b in range(a+1,N):
            if distances[a][b] == 1:
                appraisals.append(((weights[a] + weights[b])/2)**-2)
                appraisal_indexes.append([a,b])
                
    appraisal_sum = sum(appraisals)
    r = random.random()
    i = 0
    for i in range(len(appraisals)):
        a = appraisals[i] / appraisal_sum
        if r < a:
            node_a,node_b = appraisal_indexes[i]
            break
        else:
            r -= a
    # print("Chose",node_a,node_b)
    return node_a,node_b

def assignTrees(distances, node_a, node_b):
    # each node will be on either side of the chosen connection
    rowA = distances[node_a]
    rowB = distances[node_b]
    
    nA = 0 # these count the nodes in each sub-tree
    nB = 0
    
    assignments = []
    
    for i in range(N):
        if rowA[i] < rowB[i]:
            assignments.append(0)
            nA += 1
        if rowB[i] < rowA[i]:
            assignments.append(1)
            nB += 1
            
    return assignments, nA, nB

def cycleList(this_list,n):
    return this_list[n:] + this_list[:n]

###

def populate(population_size, number_positions, tree, weights, distances):
    population = []
    for i in range(population_size):
        nA,nB = 0,0
        while nB < 2:
            nodeA, nodeB = chooseConnection(weights, distances)
            assignments, nA, nB = assignTrees(distances, nodeA, nodeB)
        
        # make two list of independent random assignments within each half of the tree
        offset = random.randrange(1,nB)
        subtreeA = random.sample(range(offset, offset + nA), nA)
        subtreeB0 = random.sample(range(offset + nA, offset + N), nB)
        subtreeB = [subtreeB0[i]%N for i in range(nB)]  # wrap around the values which exceed N
        
        # put them together based on the original tree
        this_list = []
        j,k = 0,0
        for i in range(N):
            if assignments[i] == 0:
                this_list.append(subtreeA[j])
                j += 1
            if assignments[i] == 1:
                this_list.append(subtreeB[k])
                k += 1
        
        this_value,these_troublemakers = judge(this_list, tree, weights, distances)
        population.append((this_list, this_value, these_troublemakers, nodeA,nodeB, assignments, nA,nB, offset))
    return population

def repopulate(population_size, number_children, tree, weights, distances, population):
    new_population = []
    
    for i in range(number_children):
        choice = random.randint(0, population_size-1)
        kid = population[choice][0].copy()
        nodeA,nodeB, assignments, nA,nB, offset = population[choice][3:]
        print("nodeA",nodeA,"nodeB",nodeB,"offset",offset)
        
        troublemakers = list(dict.fromkeys(population[choice][2]))
        # note which subtrees have troublemakers
        bad_trees = []
        i = 0
        while i < len(troublemakers):
            trouble = kid.index(troublemakers[i])
            bad_trees.append(assignments[trouble])
            i += 1
        
        # shuffle the troublemakers
        j = 0
        while j < len(troublemakers):
            trouble = kid.index(troublemakers[j])
            rando = random.randrange(N)
            
            if assignments[trouble] == assignments[rando]:
                kid[rando],kid[trouble] = kid[trouble],kid[rando]
                j += 1
        
        # then, shuffle some more
        shuffles = random.randrange(-1, int(N / (len(troublemakers) + 1)))
        # shuffles = random.randrange(1,N)
        while shuffles > 0:
            flippers = random.sample(range(N), 2)
            rando1 = flippers[0]
            rando2 = flippers[1]
            
            if assignments[rando1] == assignments[rando2] and assignments[rando1] in bad_trees:
                kid[rando2],kid[rando1] = kid[rando1],kid[rando2]
            shuffles -= 1
            
        kid_value,kid_troublemakers = judge(kid, tree, weights, distances)
        new_population.append((kid, kid_value, kid_troublemakers, nodeA,nodeB, assignments, nA,nB, offset))
    return new_population

def getBestPop(population, new_population, J):
    all_population = population + new_population
    best_population = sorted(all_population, key = lambda x: x[1][J])
    return best_population[:len(population)]

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
    #format to 3 sig figs
    return int(number*1000)/1000

def trendAverage(trends):
    trend = []
    rows = len(trends[0])
    cols = len(trends)
    T = [[trends[j][i] for j in range(cols)] for i in range(rows)]
    for r in T:
        trend.append(sum(r)/len(r))
    return trend

def custom_tree():
    return nx.read_adjlist("tree.txt", nodetype=int)

def findSolution(N,S,J):
    nodes = N
    maxIter = 1000
    population_size = 100
    number_children = int(population_size / 2)
    
    seed = S
    tree = nx.random_tree(nodes,seed=seed)
    graph_tree(tree,1,nodes,seed)
    problem_matrix = tree_to_matrix(tree,nodes)
    print()
    print("Now on seed",S)
    #print_matrix(problem_matrix)
    weights,distances = assignWeights(problem_matrix)
    print()
    #print_matrix(distances)
    #print(weights)
    
    best1 = [(N**2) * 2 * sum(weights)] * 3  # initialize to a high value
    trend = []
    
    population = populate(population_size, nodes, problem_matrix, weights, distances)
    
    start = time.time()
    for i in range(maxIter):
        new_population = repopulate(population_size, number_children, problem_matrix, weights, distances, population)
        population = getBestPop(population, new_population, J)
        if population[0][1][J] < best1[J]:
            best0 = population[0][0]
            best1 = population[0][1]
            best2 = population[0][2]
            print("new best",best1,"on tick",i,"and it's",best0,"and the troublemakers are",best2)
        # trend.append(best1[J])
        if len(population[0][2]) == 0:
            break
        if i%50 == 0:
            print("now on tick",i)
    end = time.time()
    
    solve_time = f(end - start)
    # if len(trend) < maxIter:
    #     trend += [best1] * (maxIter - len(trend))
    
    sorted_population = sorted(population, key = lambda x: x[1])
    best_solution = sorted_population[0]
    best_value = best_solution[1][J]
    print("Best solution value:",best_solution[1][J])
    if best_value > 0:
        print("Local Optimal Solution Found!")
        id = 2
    else:
        print("Global Optimal Solution Found!")
        id = 3
    solution_matrix = reorder(listToMatrix(best_solution[0]),problem_matrix)
    print_matrix(solution_matrix)
    AL = matrix_to_AL(solution_matrix, nodes)
    solution_tree = nx.parse_adjlist(AL, nodetype=int)
    graph_tree(solution_tree,id,nodes,seed,solve_time,int(best_value))
    
    return solve_time,id,trend, best_solution[1]

if __name__ == '__main__':
    nMin = 30
    nMax = 30
    nInc = 5
    
    sMin = 0
    sMax = 1
    seeds = sMax-sMin
    seedList = [0,2,5,7,10,11,13,14,15,16,21,23,26,27,29,30,31,33,36,41,42,43,48,51,57,63,66,69,74,75,77,78,80,83,85,89,91,92,96,98]
    
    J = judgement_choice = 2 # 0=conflicts , 1=unused , 2=unused_weighted
    
    times = []
    solves = []
    Ntrends = []
    
    conflict_sum = 0
    unused = 0
    unused_weighted = 0
    
    # with open('results_splitting.csv', mode='w', newline='') as results_file:
    #     results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     results_writer.writerow(["Perfect","Time","Nodes","Seed","Evaluator","Conflicts","Unused","Unused_Weighted"])
    
    N = nMin
    while N <= nMax:
        totalTime = 0
        perfectSolves = 0
        # trends = []
        for S in seedList[sMin:sMax]:
            solveTime,id,trend, judgements = findSolution(N,S,J)
            
            totalTime += solveTime
            perfectSolves += (id-2)
            # trends.append(trend)
            
            conflict_sum += judgements[0]
            unused += judgements[1]
            unused_weighted += judgements[2]
            
            with open('results_splitting.csv', mode='a+', newline='') as results_file:
                results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                results_writer.writerow([id-2,solveTime,N,seedList.index(S),J,judgements[0],judgements[1],judgements[2]])
            
        average = f(totalTime/seeds)
        times.append(average)
        solves.append(perfectSolves)
        # Ntrend = trendAverage(trends)
        # Ntrends.append(Ntrend)
        N += nInc
    
    print()
    
    N = nMin
    i=0
    while N <= nMax:
        print("Splitting")
        print("Judgement choice:",J)
        print("Average time for",N,"nodes:",times[i],"seconds")
        print("\t Perfect solutions:",solves[i],"out of",seeds)
        print("\t Average conflicts remaining:",f(conflict_sum/seeds))
        print("\t Average unused edge labels:",f(unused/seeds))
        print("\t Average unused (weighted):",f(unused_weighted/seeds))
        N += nInc
        i += 1