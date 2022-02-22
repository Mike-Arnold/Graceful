# the backtracking method was made by Horton and improved by Fang

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import time
import csv

def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])
        
def f(number):
    return int(number*1000)/1000
        
def graph_tree(tree,id=1,nodes=20,seed=0,time=0,conflicts=0):
    # initialize the plot
    
    n_sqrt = nodes**.5
    
    plt.figure(id, figsize = (4*n_sqrt,4*n_sqrt))
    pos = graphviz_layout(tree, prog="neato")
    nx.draw(tree,pos,node_size=21000/n_sqrt)
    
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
    
    font_size = 270/n_sqrt
    nx.draw_networkx_labels(tree, pos, labels = node_labels, font_size = font_size)
    text = nx.draw_networkx_edge_labels(tree, pos, edge_labels = edge_labels, font_size = font_size)
    for _,t in text.items():
        t.set_rotation('horizontal')
    
    if id == 1:
        title = str("Problem graph \n Nodes = {} \n Seed = {}".format(nodes,seed))
    if id == 2:
        title = str("Solution graph \n Nodes = {} , Seed = {} \n Time = {} seconds".format(nodes,seed,time))
    plt.title(title,fontsize = 6*n_sqrt)
    plt.gcf().set_facecolor("#A9A9A9")
    plt.show()

def tree_to_matrix(tree,nodes):
    temp = []
    for line in nx.generate_adjlist(tree):
        my_list = line.split(" ")
        my_list.pop(0)
        my_list = [int(i) for i in my_list]
        
        temp.append(my_list)
    # fill a matrix with zeroes
    matrix = [[0]*(nodes) for i in range(nodes)]
    
    # place the 1s
    for i in range(nodes):
        for j in range(nodes):
            if j in temp[i]:
                matrix[i][j] = 1
                matrix[j][i] = 1
    return matrix

def assign_weights(matrix):
    distances = [row[:] for row in matrix]
    N = len(matrix)
    for d in range(2,N):
        for i in range(N):
            distances[i][i] = -1
            for j in range(N):
                if distances[i][j] == d-1:
                    for k in range(N):
                        if matrix[j][k] == 1 and distances[i][k] == 0:
                            distances[i][k] = d
    weights = []
    for i in range(N):
        rowsum = 0
        for j in range(N):
            if distances[i][j] == -1:
                distances[i][j] = 0
            rowsum += distances[i][j]
        weights.append(rowsum)
        
    return weights, distances

def matrix_to_edgelist(matrix,center):
    edgelist = []
    active_nodes = [center]
    
    while len(edgelist) != len(matrix) - 1:
        for a in active_nodes:
            for i in range(len(matrix)):
                b = matrix[a][i]
                if b == 1 and i not in active_nodes:
                    active_nodes.append(i)
                    edgelist.append([a,i])
    return edgelist

def reassign(edgelist,assignment):
    new_edgelist = edgelist.copy()
    
    for e in new_edgelist:
        e[0] = assignment[e[0]]
        e[1] = assignment[e[1]]
    
    return new_edgelist

def edgelist_to_tree(edgelist):
    n = len(edgelist)+1
    adjacency_list = []
    for i in range(n):
        line = str(i)
        adjacency_list.append(line)
    for e in edgelist:
        if e[0] < e[1]:
            adjacency_list[e[0]] = adjacency_list[e[0]] + " " + str(e[1])
        elif e[0] > e[1]:
            adjacency_list[e[1]] = adjacency_list[e[1]] + " " + str(e[0])
    
    tree = nx.parse_adjlist(adjacency_list, nodetype=int)
    return tree
    
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

def backtrack(value,unused,labels,labeled,active,inactive):
    if value == 0: # then out of edges to assign
        return 1
    
    # stop trying this root if too many iterations
    global T
    T += 1
    if T > 500*len(labels):
        return 0
        
    # now check if value can be applied without conflicts
    applicable = []
    for v in active:
        if labels[v[0]] + value in unused or labels[v[0]] - value in unused:
            applicable.append(v)
            
    for v in applicable:
        # print("applying",value,"to",v)
        # input()
        if labels[v[0]] + value in unused:
            new_label = labels[v[0]] + value
        elif labels[v[0]] - value in unused:
            new_label = labels[v[0]] - value
        else:
            next(applicable, None)
        #                                                                   do
        old_label = labels[v[1]] # in case of failure
        # apply the new label to v
        labels[v[1]] = new_label
        unused.remove(new_label)
        # move v from active to labeled
        labeled.append(v)
        active.remove(v)
        # add to active based on v[1]
        movers = []
        for w in inactive:
            if w[0] == v[1]:
                movers.append(w)
        for w in movers:
            active.append(w)
            inactive.remove(w)
        
        #                                                                  try
        if backtrack(value-1,unused,labels,labeled,active,inactive):
            return 1
        else: #                                                           undo
            # print("removing",value,"from",v)
            # input()
            active.append(v)
            labeled.remove(v)
            unused.append(new_label)
            labels[v[1]] = old_label
            movers = []
            for w in active:
                if w[0] == v[1]:
                    movers.append(w)
            for w in movers:
                inactive.append(w)
                active.remove(w)
    return 0

def solveNS(N,S):
    # initializing
    tree = nx.random_tree(N,seed=S)
    # graph_tree(tree,1,N,S)
    
    start = time.time()
    
    matrix = tree_to_matrix(tree,N)
    weights, distances = assign_weights(matrix)
    center = weights.index(min(weights))
    
    # build a list of possible roots to try, starting with the center
    # and then add the nodes that are 2, 4, etc distance away from it
    roots = [center]
    this_even_number = 2
    while this_even_number < N:
        for i in range(N):
            if distances[center][i] == this_even_number:
                roots.append(i)
        this_even_number += 2
    
    for root in roots:
        print("trying root",root)
        global T
        T = 0
        # more initializing for each root
        edgelist = matrix_to_edgelist(matrix,root)
        unused = list(range(1,N)) # unused edge labels
        labels = [0]*N # labels[i] will be the label for node i
        labeled,active,inactive = [],[],[]
        for e in edgelist:
            if e[0] == root:
                active.append(e)
            else:
                inactive.append(e)
        value = max(unused)
        
        result = backtrack(value,unused,labels,labeled,active,inactive)
        if result == 1:
            break
    
    end = time.time()
    solve_time = end-start
    
    # # format solution
    # new_edgelist = reassign(edgelist,labels)
    # new_tree = edgelist_to_tree(new_edgelist)
    # seedList = [0,2,5,7,10,11,13,14,15,16,21,23,26,27,29,30,31,33,36,41,42,43,48,51,57,63,66,69,74,75,77,78,80,83,85,89,91,92,96,98]
    # graph_tree(new_tree,2,N,seedList.index(S),f(solve_time))
    
    return result,solve_time

if __name__ == '__main__':
    # with open('results_backtrack.csv', mode='w', newline='') as results_file:
    #     results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     results_writer.writerow(["Perfect","Time","Nodes","Seed"])
    
    Nmin = 100
    Nmax = 100
    Ninc = 5
    
    Smin = 0
    Smax = 1
    seedList = [0,2,5,7,10,11,13,14,15,16,21,23,26,27,29,30,31,33,36,41,42,43,48,51,57,63,66,69,74,75,77,78,80,83,85,89,91,92,96,98]
    
    N = Nmin
    while N <= Nmax:
        total_time = 0
        S = Smin
        while S < Smax:
            T = 0
            result,solve_time = solveNS(N,seedList[S])
            total_time += solve_time
            # with open('results_backtrack.csv', mode='a+', newline='') as results_file:
            #     results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     results_writer.writerow([result,solve_time,N,S])
            if result:
                print("N =",N,"S =",S,"Successful solve in",f(solve_time),"on iteration",T)
            else:
                print("N =",N,"S =",S,"Non-solve","on iteration",T)
            S += 1
        print("Average time for",N,"nodes:",f(total_time/(Smax-Smin)))
        N += Ninc