import sys
import json
import time
from operator import add
from pyspark import SparkContext

review_filepath = sys.argv[1]
business_filepath = sys.argv[2]
output_filepath_question_a = sys.argv[3]
output_filepath_question_b = sys.argv[4]

ansB = {}
sc = SparkContext.getOrCreate()

# A

reviewA = sc.textFile(review_filepath).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['stars']))\
    .filter(lambda x: x[1] is not None).groupByKey().map(lambda x: (x[0], (sum(x[1]), len(x[1]))))

businessA = sc.textFile(business_filepath).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['city']))\
    .map(lambda x: (x[0], "") if x[1] is None else (x[0], x[1])).sortBy(lambda x: x[1])

A = businessA.leftOuterJoin(reviewA).map(lambda x: x[1]).filter(lambda x: x[1] is not None)\
    .reduceByKey(lambda x, y: (x[0]+y[0],x[1]+y[1])).mapValues(lambda x: float(x[0]/x[1]))
ansA = A.sortBy(lambda x: (-x[1], x[0])).collect()
with open(output_filepath_question_a, 'w+') as output:
    output.write('city,stars' + '\r')
    for _, kv in enumerate(ansA):
        line = kv[0] + ',' + str(kv[1]) + '\r'
        output.write(line)

# B method1

start1 = time.clock()
ansB1 = sorted(A.collect(), key=lambda x: (-x[1], x[0]))
print(ansB1[:10])
end1 = time.clock()
ansB["m1"] = end1 - start1

# B method2

start2 = time.clock()
ansB2 = A.sortBy(lambda x: (-x[1], x[0])).take(10)
print(ansB2)
end2 = time.clock()
ansB["m2"] = end2 - start2
ansB["reason"] = "When sorting in python, data is sorted in ram, and it's quicker than the distributed sorting."

with open(output_filepath_question_b, 'w+') as output:
    json.dump(ansB, output)
output.close()
