import sys
import math
from time import time
from itertools import combinations, permutations
from collections import defaultdict, deque, OrderedDict
from pyspark import SparkContext


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.btwns = defaultdict(float)
        self.communities = list()

    def add_edge(self, v1, v2):
        self.graph[v1].append(v2)

    def remove_edge(self, v1, v2):
        self.graph[v1].remove(v2)

    def get_graph(self):
        return self.graph

    def add_btwns(self, v_pairs, score):
        self.btwns[v_pairs] = score

    def update_btwns(self, v_pairs, score):
        self.btwns[v_pairs] += score

    def remove_btwns(self):
        self.btwns = defaultdict(float)

    def get_btwns(self, v_pairs):
        return self.btwns.get(v_pairs)

    def get_btw_items(self):
        return self.btwns.items()

    def update_communities(self, best_list):
        self.communities = best_list

    def get_communities(self):
        return self.communities


def calculate_btwns(g, idx):
    visited = dict()
    adj = g.get_graph()
    dist = dict()
    paths = dict()

    q = deque()
    q.append(idx)
    visited[idx] = True
    dist[idx] = 0
    paths[idx] = 1

    parent = defaultdict(set)
    child = defaultdict(set)
    while q:
        cur = q[0]
        q.popleft()
        for nei in adj[cur]:

            if not dist.get(nei):
                dist[nei] = math.inf
                paths[nei] = 0

            if not visited.get(nei):
                q.append(nei)
                visited[nei] = True
            if dist[nei] > dist[cur] + 1:
                dist[nei] = dist[cur] + 1
                dist[idx] = 0
                paths[nei] = paths[cur]
            elif dist[nei] == dist[cur] + 1:
                paths[nei] += paths[cur]
            if dist[nei] > dist[cur]:
                child[cur].add(nei)
            elif dist[nei] < dist[cur]:
                parent[cur].add(nei)

    lv_dict = OrderedDict(sorted(dist.items(), key=lambda x: -x[1]))
    bottom = next(iter(lv_dict.values()))
    n_score = dict()
    btwns_score = dict()
    for k, v in lv_dict.items():
        if bottom == 0:
            break
        if v == bottom or (not child.get(k)):
            n_score[k] = 1
            for p in parent[k]:
                sk = tuple(sorted([k, p]))
                if btwns_score.get(sk):
                    btwns_score[sk] += 1 / paths[k] * paths[p]
                else:
                    btwns_score[sk] = 1 / paths[k] * paths[p]
        elif v == 0:
            break
        else:
            n_score[k] = 1
            for c in child[k]:
                sk = tuple(sorted([k, c]))
                n_score[k] += btwns_score[sk]
            for p in parent[k]:
                sk = tuple(sorted([k, p]))
                if btwns_score.get(sk):
                    btwns_score[sk] += n_score[k] / paths[k] * paths[p]
                else:
                    btwns_score[sk] = n_score[k] / paths[k] * paths[p]
    for k, v in btwns_score.items():
        if g.get_btwns(k):
            g.update_btwns(k, 0.5 * v)
        else:
            g.add_btwns(k, 0.5 * v)
    return g


start = time()

#filter_threshold = int(sys.argv[1])
#input_file_path = sys.argv[2]
#btwns_output_file_path = sys.argv[3]
#community_output_file_path = sys.argv[4]
filter_threshold = 7
input_file_path = "ub_sample_data.csv"
btwns_output_file_path = "2_1.txt"
community_output_file_path = "2_2.txt"
sc = SparkContext.getOrCreate()
ub_rdd = sc.textFile(input_file_path).filter(lambda x: x != "user_id,business_id") \
    .map(lambda x: (x.split(',')[0], [x.split(',')[1]])) \
    .reduceByKey(lambda x, y: x + y).map(lambda x: (x[0], list(set(x[1])))) \
    .filter(lambda x: len(x[1]) >= filter_threshold)
ub_dict = ub_rdd.collectAsMap()
u_list = ub_rdd.map(lambda x: x[0]).collect()
u_pairs = combinations(u_list, 2)

edges_list = list()
vertices_set = set()
g = Graph()
n = 0
for pair in u_pairs:
    common_num = len(set(ub_dict[pair[0]]).intersection(set(ub_dict[pair[1]])))
    if common_num >= filter_threshold:
        n += 1
        g.add_edge(pair[0], pair[1])
        g.add_edge(pair[1], pair[0])
        vertices_set.add((pair[0],))
        vertices_set.add((pair[1],))

vertices_list = list(vertices_set)

for vertex in vertices_list:
    ver = vertex[0]
    g = calculate_btwns(g, ver)

btwns_list = sorted(list(g.get_btw_items()), key=lambda x: (-x[1], x[0]))


# task2_2

ori_adj = g.get_graph()
cur_g = g
cur_max = 0
edges_left = n
while True:
    cur_adj = cur_g.get_graph()
    visited = dict()
    communities = []
    for v in vertices_list:

        if visited.get(v[0]):
            continue
        temp_graph = []
        q = deque()
        q.append(v[0])
        visited[v[0]] = True
        temp_graph.append(v[0])
        while q:
            curr = q[0]
            q.popleft()
            for neighbor in cur_adj[curr]:
                if not visited.get(neighbor):
                    q.append(neighbor)
                    visited[neighbor] = True
                    temp_graph.append(neighbor)
        communities.append(sorted(temp_graph))

    modularity = 0
    for community in communities:
        modularity_part = 0
        for v_pair in list(permutations(community, 2)):

            if v_pair[1] in ori_adj.get(v_pair[0]):
                A_ij = 1
            else:
                A_ij = 0

            ki = len(ori_adj.get(v_pair[0]))
            kj = len(ori_adj.get(v_pair[1]))

            modularity_part += (A_ij - ki * kj / (2 * n))
        modularity += modularity_part
    modularity = 1 / (2 * n) * modularity
    if modularity > cur_max:
        cur_max = modularity
        g.update_communities(communities)
    btw_dict = OrderedDict(sorted(cur_g.get_btw_items(), key=lambda x: -x[1]))
    cut_edges = []
    highest_score = next(iter(btw_dict.values()))
    for key, val in btw_dict.items():
        if val == highest_score:
            cut_edges.append(key)
        else:
            break
    edges_left -= len(cut_edges)
    if edges_left == 0:
        break
    for edge in cut_edges:
        try:
            cur_g.removeEdge(edge[0], edge[1])
            cur_g.removeEdge(edge[1], edge[0])
        except:
            continue
    cur_g.remove_btwns()
    for vertex in vertices_list:
        cur_g = calculate_btwns(cur_g, vertex[0])

communities_list = g.get_communities()
for i in range(len(communities_list)):
    communities_list[i]=sorted(communities_list[i], key=str.lower)
    print(communities_list[i])

communities_list = sorted(communities_list, key=lambda x: (len(x), x))
with open(btwns_output_file_path, "w+") as output_file:
    for x in btwns_list:
        output_file.write(str(x)[1:-1] + '\n')
output_file.close()

with open(community_output_file_path, "w+") as output_file:
    for x in communities_list:
        output_file.write(str(x)[1:-1] + '\n')
output_file.close()

end = time()
print(end - start)
