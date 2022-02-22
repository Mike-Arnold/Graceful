from docplex.mp.model import Model
from docplex.mp.environment import Environment
import itertools
import time
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import csv

# ----------------------------------------------------------------------------
# Initialize the Data
# ----------------------------------------------------------------------------
def generate_random_tree(nodes, seed = None):
    return nx.random_tree(nodes,seed=seed)

# ----------------------------------------------------------------------------
# Build the Model
# ----------------------------------------------------------------------------
def build_graceful_labeling_problem(matrix,n):
    nodes = list(range(n))
    
    # ----------------------------------------------------------------------------
    # Create Model Environment
    # ----------------------------------------------------------------------------
    mdl = Model(name = 'Graceful Labeling Problem')
    
    # ----------------------------------------------------------------------------
    # Create Decision Variables
    # ----------------------------------------------------------------------------     
    mdl.order = mdl.binary_var_dict(list(itertools.product(nodes,nodes)), name = 'var_order')
    mdl.result = mdl.binary_var_dict(list(itertools.product(nodes,nodes)), name = 'var_result')
    
   
    # ----------------------------------------------------------------------------
    # Create Constraints
    # ----------------------------------------------------------------------------
    # each row and column of the order matrix has a single 1
    mdl.add((mdl.sum(mdl.order[i,j] for i in nodes) <= 1) for j in nodes)
    mdl.add((mdl.sum(mdl.order[i,j] for j in nodes) >= 1) for i in nodes)
    
    strategy = 1  # 1 = no objective | 2 = conflicts | 3 = unused | 4 = unused weighted
    
    if strategy == 1:
        # each diagonal has a single 1
        # simple constraint
        mdl.add((mdl.sum(mdl.result[j-i,j+1] for j in list(range(i,nodes[-1])) ) <= 1) for i in nodes)
        mdl.minimize(0)
    
    if strategy == 2:
        # track conflicts to minimize them
        mdl.conflict = mdl.integer_var_dict((i for i in nodes), name = 'conflict', lb = 1)  # note the lower bound is 1
        mdl.add((mdl.sum(mdl.result[j-i,j+1] for j in nodes[i:-1] ) <= mdl.conflict[i]) for i in nodes)  # note the <=
        mdl.minimize(mdl.sum(mdl.conflict[i] for i in nodes) - n)
    
    if strategy == 3:
        # track the number of unused
        mdl.unused = mdl.integer_var_dict((i for i in nodes), name = 'unused', ub = 1, lb = 0)
        mdl.add((mdl.sum(mdl.result[j-i,j+1] for j in nodes[i:-1] ) >= 1 - mdl.unused[i]) for i in nodes)
        mdl.add(mdl.result[j-i,j+1] <= 1 - mdl.unused[i] for i in nodes for j in nodes[i:-1])
        mdl.minimize(mdl.sum(mdl.unused[i] for i in nodes))
        
    if strategy == 4:
        # track the weighted number of unused, where bigger edge labels elicit a bigger penalty
        # judge the number of unused one of in two different ways
        mdl.unused = mdl.integer_var_dict((i for i in nodes), name = 'unused', ub = 1, lb = 0)
        mdl.add((mdl.sum(mdl.result[j-i,j+1] for j in nodes[i:-1] ) >= 1 - mdl.unused[i]) for i in nodes)
        mdl.add(mdl.result[j-i,j+1] <= 1 - mdl.unused[i] for i in nodes for j in nodes[i:-1])
        mdl.minimize(mdl.sum(mdl.unused[i] * (i+1) for i in nodes))
    
    # some fast constraints which rule out obvious wrong answers
    # middle diagonal is all zeroes
    mdl.add(mdl.result[i,i] == 0 for i in nodes)
    # result is symmetric
    mdl.add(mdl.result[i,j] == mdl.result[j,i] for i in nodes for j in nodes)
    # same number of edges
    mdl.add(mdl.sum(mdl.result[i,j] - matrix[i][j] for i in nodes for j in nodes) == 0)
    
    
    # preserve edges - the resulting tree is the same as the original tree
    # moderate O(n3) constraint
    mdl.add(mdl.result[i,j] - mdl.sum(mdl.order[i,r]*matrix[r][s] for r in nodes) <= 1 - mdl.order[j,s]
                        for i in nodes for j in nodes for s in nodes)
    mdl.add(mdl.sum(mdl.order[j,s]*matrix[r][s] for s in nodes) - mdl.result[i,j] <= 1 - mdl.order[i,r]
                        for i in nodes for j in nodes for r in nodes)
    
    
    return mdl

# ----------------------------------------------------------------------------
# Solve the Model
# ----------------------------------------------------------------------------
def solve_model(mdl, log_output = None, time_limit = None):
    print("\n\n===============================Solving Problem===============================\n")
    mdl.set_time_limit(time_limit)
    mdl.set_log_output(log_output)

    return mdl.solve()

# ----------------------------------------------------------------------------
# Format the Solution Obtained
# ----------------------------------------------------------------------------
def format_solution(model, nodes):
    print("\n\n===================================Solution==================================\n")   
    
    solution_matrix = []
    adjacency_list = []
    for i in range(nodes):
        temp = []
        temp1 = str(i)
        for j in range(nodes):
            foo = int(model.result[i,j].solution_value)
            #foo = int(model.order[i,j].solution_value)
            temp.append(foo)
            if(j>i and foo == 1):
                temp1 += " " + str(j)
        solution_matrix.append(temp)
        adjacency_list.append(temp1)
    return solution_matrix,adjacency_list

def diagnostics(model,nodes):
    node_list = list(range(nodes))

    conflicts = []
    used = []
    for i in range(nodes):
        these_conflicts = 0
        this_used = 0
        for j in node_list[i:-1]:
            if int(model.result[j-i,j+1].solution_value) > 0:
                these_conflicts += int(model.result[j-i,j+1].solution_value)
                this_used += int(model.result[j-i,j+1].solution_value)
        conflicts.append(max(these_conflicts,1))
        used.append(this_used)
        
    conflict_sum = sum(conflicts) - nodes
    
    unused = []
    unused_weighted = 0
    for i in range(nodes-1):
        if used[i] == 0:
            unused.append(1)
            unused_weighted += i+1
        else:
            unused.append(0)
            
    unused_sum = sum(unused)
    
    return conflict_sum,unused_sum,unused_weighted

# ----------------------------------------------------------------------------
# Make a custom tree
# ----------------------------------------------------------------------------
def custom_tree():
    return nx.read_adjlist("tree.txt", nodetype=int)

# ----------------------------------------------------------------------------
# Graph the Tree
# ----------------------------------------------------------------------------
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
        title = str("Local solution with {} conflicts \n Nodes = {} \n Seed = {} \n Time to solve: {} seconds".format(conflicts,nodes,seed,time))
    #    file_name = str("Solution_graph_{}_{}.png".format(nodes,seed))
    if id == 3:
        title = str("Global solution \n Nodes = {} \n Seed = {} \n Time to solve: {} seconds".format(nodes,seed,time))
    #    file_name = str("Solution_graph_{}_{}.png".format(nodes,seed))
    plt.title(title,fontsize = font_size)
    #my_path = os.path.dirname(__file__)
    plt.gcf().set_facecolor("#A9A9A9")
    #plt.savefig(my_path + '\\graphs2\\' + file_name, bbox_inches='tight')
    plt.show()
    
# ----------------------------------------------------------------------------
# Make adjacency matrix
# ----------------------------------------------------------------------------
def make_matrix(tree,nodes):
    temp = []
    for line in nx.generate_adjlist(tree):
        #print(line)
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
        
# ----------------------------------------------------------------------------
# Print matrix
# ----------------------------------------------------------------------------
def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])
        
def f(number):
    #format to 3 sig figs
    return int(number*1000)/1000
    
# ----------------------------------------------------------------------------
# Main Program
# ----------------------------------------------------------------------------    
if __name__ == '__main__':
    print("===================================Environment==================================\n")
    env = Environment()
    env.print_information()
    
    # with open('results_cplex.csv', mode='w', newline='') as results_file:
    #     results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     results_writer.writerow(["Perfect","Time","Nodes","Seed","Evaluator","Conflicts","Unused","Unused_Weighted"])
    
    '''Generate the Tree'''
    average_times,average_conflicts,average_unused,average_unused_weighted = [],[],[],[]
    a,b = 2,7
    for i in range(a,b):
        nodes = 5*i
        total_time,total_conflicts,total_unused,total_unused_weighted = 0,0,0,0
        seedList = [0,2,5,7,10,11,13,14,15,16,21,23,26,27,29,30,31,33,36,41,42,43,48,51,57,63,66,69,74,75,77,78,80,83,85,89,91,92,96,98]
        c,d = 0,1
        for seed in seedList[c:d]:
            tree = generate_random_tree(nodes,seed=seed)
            
            '''Graph Tree'''
            graph_tree(tree,1,nodes,seed)
            problem_matrix = make_matrix(tree,nodes)
            # print("==================================Starting Tree=================================\n")
            # print_matrix(problem_matrix)
            
            '''Build the Model'''
            model = build_graceful_labeling_problem(problem_matrix,nodes)
            
            '''Solve the Model'''
            start = time.time()
            solution = solve_model(model, log_output = True, time_limit = 3600)
            end = time.time()
            solve_time = f(end - start)
            total_time += solve_time
            print("\n\nTime to Solve the Model: {}".format(solve_time))
            
            '''Print and Plot Solution'''
            if solution:
                solution_matrix, adjacency_list = format_solution(model, nodes)
                print_matrix(solution_matrix)

                conflicts,unused,unused_weighted = diagnostics(model,nodes)
                print("Conflicts:",conflicts)
                print("Unused:",unused)
                print("Unused (Weighted):",unused_weighted)
                
                total_conflicts += conflicts
                total_unused += unused
                total_unused_weighted += unused_weighted
                
                with open('results_cplex.csv', mode='a+', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow([1,solve_time,nodes,seedList.index(seed),0,conflicts,unused,unused_weighted])

                tree2 = nx.parse_adjlist(adjacency_list, nodetype=int)
                graph_tree(tree2,2,nodes,seed,solve_time)
                
            else:
                print("Problem has no Solution")
                conflicts,unused,unused_weighted = diagnostics(model,nodes)
                print("Conflicts:",conflicts)
                print("Unused:",unused)
                print("Unused (Weighted):",unused_weighted)
                
                with open('results_cplex.csv', mode='a+', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow([1,solve_time,nodes,seedList.index(seed),0,-1,-1,-1])

        average_times.append(total_time/(d-c))
        average_conflicts.append(total_conflicts/(d-c))
        average_unused.append(total_unused/(d-c))
        average_unused_weighted.append(total_unused_weighted/(d-c))
        
    for i in range(a,b):
        print("Average time for",5*i,"nodes:",f(average_times[i-a]))