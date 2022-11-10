import sys
import json
import time
from operator import add
from pyspark import SparkContext

review_filepath = sys.argv[1]
output_filepath = sys.argv[2]
n = int(sys.argv[3])

ans = {}
ans["default"] = {}
ans["customized"] = {}
sc = SparkContext.getOrCreate()
lines = sc.textFile(review_filepath).map(lambda x: json.loads(x)).cache()

# default
start1 = time.time()
lines1 = lines.map(lambda x: (x["business_id"], 1))
res1 = lines1.reduceByKey(add).sortBy(lambda x: (-x[1], x[0])).take(10)
end1 = time.time()
ans["default"]["n_partition"] = lines1.getNumPartitions()
ans["default"]["n_items"] = lines1.glom().map(len).collect()
ans["default"]["exe_time"] = end1 - start1

# customized
start2 = time.time()
lines2 = lines.map(lambda x: (x["business_id"], 1)).partitionBy(n, lambda x: ord(x[0][0]))
res2 = lines2.reduceByKey(add).sortBy(lambda x: (-x[1], x[0])).take(10)
end2 = time.time()
ans["customized"]["n_partition"] = lines2.getNumPartitions()
ans["customized"]["n_items"] = lines2.glom().map(len).collect()
ans["customized"]["exe_time"] = end2 - start2

with open(output_filepath, 'w+') as output:
    json.dump(ans, output)
output.close()
