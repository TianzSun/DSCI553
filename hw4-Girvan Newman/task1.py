import sys
import os
from time import time
from itertools import combinations
from pyspark import SparkContext
from pyspark.sql import SparkSession
from graphframes import GraphFrame

start = time()
os.environ["PYSPARK_SUBMIT_ARGS"] = (
"--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")


filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]

sc = SparkContext.getOrCreate()
sparkSession = SparkSession(sc)
ub_rdd = sc.textFile(input_file_path).filter(lambda x: x != "user_id,business_id")\
    .map(lambda x: (x.split(',')[0], [x.split(',')[1]]))\
    .reduceByKey(lambda x, y: x+y).map(lambda x: (x[0], list(set(x[1]))))\
    .filter(lambda x: len(x[1]) >= filter_threshold)
ub_dict = ub_rdd.collectAsMap()
u_list = ub_rdd.map(lambda x: x[0]).collect()
u_pairs = combinations(u_list, 2)

edges_list = list()
vertices_set = set()
for pair in u_pairs:
    common_num = len(set(ub_dict[pair[0]]).intersection(set(ub_dict[pair[1]])))
    if common_num >= filter_threshold:
        edges_list.append(tuple(pair))
        edges_list.append(tuple([pair[1], pair[0]]))
        vertices_set.add((pair[0],))
        vertices_set.add((pair[1],))
vertices_df = sc.parallelize(list(vertices_set)).toDF(['id'])
edges_df = sc.parallelize(edges_list).toDF(["src", "dst"])
graph_frame = GraphFrame(vertices_df, edges_df)

communities = graph_frame.labelPropagation(maxIter=5)
communities_list = communities.rdd.map(lambda x: (x[1], [x[0]]))\
    .reduceByKey(lambda x, y: x+y).map(lambda x: sorted(x[1]))\
    .sortBy(lambda x: (len(x), x)).collect()

with open(community_output_file_path, 'w+') as output_file:
    for x in communities_list:
        output_file.writelines(str(x)[1:-1] + "\n")
    output_file.close()

end = time()
print(end-start)
