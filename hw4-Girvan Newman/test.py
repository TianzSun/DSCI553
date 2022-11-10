from pyspark import SparkContext
from pyspark import SparkConf
import sys
import time
from itertools import combinations, permutations
from collections import defaultdict, deque, OrderedDict
import math

# spark settings
def CreateSparkContext():
    sConf = SparkConf().setMaster("local[3]") \
            .setAppName('task2') \
            .set("spark.ui.showConsoleProgress", "false") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=sConf)
    sc.setLogLevel("ERROR")

    return sc

# Building a graph class
# ref: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
class Graph:
    # constructor
    def __init__(self):
        # dict to save graph
        # append adjacency list here
        # {vertex1: [vertices connect with it]}
        self.graph = defaultdict(list)
        # dict to save the number of shortest paths for each edge
        # {(vertex1, vertex2): num}
        self.betweenness = defaultdict(float)
        # best communities
        self.communities = list()

    # add an edge to graph
    def addEdge(self, v1, v2):
        self.graph[v1].append(v2)

    def removeEdge(self, v1, v2):
        self.graph[v1].remove(v2)

    def getGraph(self):
        return self.graph

    def newBetweenness(self, v_pairs, score):
        self.betweenness[v_pairs] = score

    def updateBetweenness(self, v_pairs, score):
        self.betweenness[v_pairs] += score

    def removeBetweenness(self):
        self.betweenness = defaultdict(float)

    def getBetweenness(self, v_pairs):
        return self.betweenness.get(v_pairs)

    def get_btw_items(self):
        return self.betweenness.items()

    def updateCommunities(self, best_lst):
        self.communities = best_lst

    def getCommunities(self):
        return self.communities

# Girvan-Newman Algorithm
# input a graph and source node
# step 1: do BFS, find number of shortest paths as label of each node,
#            save level tree, save parent and child relationship
# step 2: from bottom, calculate edge betweeness
#
# ref: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
# ref: https://www.baeldung.com/cs/graph-number-of-shortest-paths
# ref: https://www.analyticsvidhya.com/blog/2020/04/community-detection-graphs-networks/
def computeBetweenness(g, src):
    # BFS from the root
    visited = dict() # for BFS
    adj = g.get_graph() # get adjacent lists
    dist = dict() # distance/level
    paths = dict() # number of shortest paths

    q = deque() # initialize queue for BFS
    q.append(src) # append root to queue
    visited[src] = True # visited root
    dist[src] = 0 # level 0/root
    paths[src] = 1 # label as 1

    parent = defaultdict(set)
    child = defaultdict(set)
    while q: # while q not empty
        curr = q[0]
        q.popleft()
        # for each node connected to curr
        for neighbor in adj[curr]:
            # if not yet initialized, initialize
            if not dist.get(neighbor):
                dist[neighbor] = math.inf # -> infinite
                paths[neighbor] = 0
            # if not yet visited, push it to the queue, visited.
            if not visited.get(neighbor):
                q.append(neighbor)
                visited[neighbor] = True
            # check if the neighbor is a child of current node
            # its parent or same level node should not change
            # if a child is already visited (dist updated)
            # more than 1 shortest path occured
            if dist[neighbor] > dist[curr] + 1:
                dist[neighbor] = dist[curr] + 1
                dist[src] = 0 # make sure source node is at level 0
                paths[neighbor] = paths[curr]
            elif dist[neighbor] == dist[curr] + 1:
                paths[neighbor] += paths[curr]
            # update child and parent relationship by its dist
            if dist[neighbor] > dist[curr]:
                child[curr].add(neighbor)
            elif dist[neighbor] < dist[curr]:
                parent[curr].add(neighbor)
            #print(dist)
    #print(paths)
    # print(child)
    # sort by level, descending
    level_dict = OrderedDict(sorted(dist.items(), key=lambda x: -x[1]))
    bottom = next(iter(level_dict.values()))
    node_score = dict() # for this tree only
    betweenness_score = dict() # for this tree only
    for key, val in level_dict.items():
        if bottom == 0:
            #print('single node ?')
            break
        if val == bottom or (not child.get(key)): # leaf gets credit 1
            node_score[key] = 1
            # for each parent, add betweenness to edge
            for p in parent[key]:
                search_key = tuple(sorted([key, p]))
                if betweenness_score.get(search_key):
                    betweenness_score[search_key] += 1/paths[key]*paths[p]
                else:
                    betweenness_score[search_key] = 1/paths[key]*paths[p]
        elif val == 0: # at root
            break
        else:
            # look for child, add them up betweenness and self
            node_score[key] = 1 # self
            for c in child[key]:
                search_key = tuple(sorted([key, c]))
                node_score[key] += betweenness_score[search_key]
            for p in parent[key]:
                search_key = tuple(sorted([key, p]))
                if betweenness_score.get(search_key):
                    betweenness_score[search_key] += node_score[key]/paths[key]*paths[p]
                else:
                    betweenness_score[search_key] = node_score[key]/paths[key]*paths[p]
    #print(betweenness_score)
    # update to graph # betweenness divided by 2
    for key, val in betweenness_score.items():
        if g.calculate_betweenness(key):
            g.update_betweenness(key, 0.5 * val)
        else:
            g.add_betweenness(key, 0.5 * val)
    return g

def clustering(g, v_lst):
    adj = g.get_graph()
    visited = dict()
    graphs = []
    for vertex in v_lst:
        # if visited, already in a graph, pass
        if visited.get(vertex[0]):
            continue
        temp_graph = []
        q = deque() # initialize queue for BFS
        q.append(vertex[0]) # append root to queue
        visited[vertex[0]] = True # visited current root
        temp_graph.append(vertex[0])
        while q:
            curr = q[0]
            q.popleft()
            for neighbor in adj[curr]:
                if not visited.get(neighbor):
                    q.append(neighbor)
                    visited[neighbor] = True
                    temp_graph.append(neighbor)
        graphs.append(sorted(temp_graph))

    return graphs

def computeModularity(community, adj, m):
    result = 0 # initialize
    for v_pair in list(permutations(community, 2)):
        # get A_ij
        if v_pair[1] in adj.get(v_pair[0]):
            A_ij = 1
        else:
            A_ij = 0
        # get ki,kj
        ki = len(adj.get(v_pair[0]))
        kj = len(adj.get(v_pair[1]))
        # calculate
        result += (A_ij - ki*kj/(2*m))
    return result

def findCommunities(orig_g, adj, m, v_lst):
    curr_g = orig_g
    curr_max = -1
    edges_left = m
    while True:
        # detect existing communities
        communities = clustering(curr_g, v_lst)
        # compute modularity
        modularity = 0 # initialize
        for community in communities:
            this_modularity = computeModularity(community, adj, m)
            modularity += this_modularity
        modularity = 1/(2*m)*modularity
        # compare and update modularity
        if modularity > curr_max:
            curr_max = modularity
            # update on original graph, cut on current temporary graph
            orig_g.update_communities(communities)
        # Find and cut highest score edge
        btw_dict = OrderedDict(sorted(curr_g.get_btw_items(), key=lambda x: -x[1]))
        cut_edges = []
        highest_score = next(iter(btw_dict.values()))
        for key, val in btw_dict.items():
            if val == highest_score:
                cut_edges.append(key)
            else:
                break
        #print(cut_edges)
        edges_left -= len(cut_edges)
        if edges_left == 0:
            break
        # Why ValueError??
        for edge in cut_edges:
            try:
                curr_g.removeEdge(edge[0], edge[1])
                curr_g.removeEdge(edge[1], edge[0])
            except:
                continue
        curr_g.removeBetweenness()
        for vertex in v_lst:
            curr_g = computeBetweenness(curr_g, vertex[0])
        #print(orig_g.get_btw_items())
    return orig_g


if __name__ == '__main__':

    start = time.time()

    sc= CreateSparkContext()
    threshold = int(sys.argv[1])
    input_path = sys.argv[2]
    output1_path = sys.argv[3]
    output2_path = sys.argv[4]

    '''
    similar implementation from task1.py
    except for using Graph()
    '''
    # read input file, remove header
    # group and reduce by userid
    # remove users with ratings less than threshold (reduce size)
    ub_rdd = sc.textFile(input_path)
    header = ub_rdd.first()
    user_busi_rdd = ub_rdd.filter(lambda x: x!=header) \
                        .map(lambda x: (x.split(',')[0], [x.split(',')[1]])) \
                        .reduceByKey(lambda a,b: a+b) \
                        .map(lambda x: (x[0], list(set(x[1])))) \
                        .filter(lambda x: len(x[1]) >= threshold)
    # get the list of users for doing combinations
    user_list = user_busi_rdd.map(lambda x: x[0]).collect()
    # get dict for searching
    user_busi_dict = user_busi_rdd.collectAsMap()

    # for each combination check intersection len >= threshold
    # if true, then save the pair to the graph
    # and save each node to the list of vertices
    # Creating graph
    g = Graph()
    vertices_lst = []
    num_of_edges = 0
    for comb in list(combinations(user_list, 2)):
        if len(set(user_busi_dict[comb[0]]).intersection(set(user_busi_dict[comb[1]]))) >= threshold:
            g.addEdge(comb[0], comb[1])
            # Thoughts from Piazza discussion: save the reverse pair to indicate an undirected graph
            g.addEdge(comb[1], comb[0])
            num_of_edges += 1
            # make it tuple! for createDataFrame(), str doesn't work
            vertices_lst.append((comb[0],))
            vertices_lst.append((comb[1],))
    # remove duplicate vertices, no duplicate edges since distinct combs.
    vertices_lst = list(set(vertices_lst))

    # do GN Algorithm for each vertex
    for vertex in vertices_lst:
        g = computeBetweenness(g, vertex[0])

    betweenness_output = sorted(list(g.get_btw_items()), key=lambda x: (-x[1], x[0]))
    with open(output1_path, "w") as fp:
        for item in betweenness_output:
            # remove outer parenthesis
            fp.write(str(item)[1:-1]+'\n')
    fp.close()

    # get original graph adjacent lists
    original_adj = g.getGraph()
    # try to return a complete graph with updated best communities
    final_g = findCommunities(g, original_adj, num_of_edges, vertices_lst)
    comm = final_g.get_communities()
    comm_output = sorted(comm, key=lambda x: (len(x), x))
    with open(output2_path, "w") as fp:
        for item in comm_output:
            # remove outer []
            fp.write(str(item)[1:-1]+'\n')
    fp.close()

    end = time.time()
    print("Duration: "+str(end-start))
