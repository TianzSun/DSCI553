import sys
import math
import random
from time import time
from itertools import combinations
from pyspark import SparkContext

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]



def get_bands(index_list, band_num):
    res = []
    r = math.ceil(len(index_list) / band_num)
    for i in range(0, len(index_list), r):
        res.append(tuple(sorted(index_list[i:i+r])))
    return res


def jac(set1, set2):
    return float(len(set1 & set2)) / float(len(set1 | set2))


def get_hash_fun(a, b, m):
    def hash_fun(x):
        return ((a*x + b) % 10000001) % m
    return hash_fun


start = time()

sc = SparkContext.getOrCreate()

raw = sc.textFile(input_file_name).filter(lambda x: x != "user_id,business_id,stars")

user_idx_rdd = raw.map(lambda x: x.split(',')[0]).distinct().sortBy(lambda x: x).zipWithIndex()
user_idx_dict = user_idx_rdd.collectAsMap()
business_idx_dict = raw.map(lambda x: x.split(',')[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

hash_function_list = []

a_list = random.sample(range(1, 1000000), 20)
b_list = random.sample(range(1000000), 20)

for a, b in zip(a_list, b_list):
    hash_function = get_hash_fun(a, b, len(user_idx_dict))
    hash_function_list.append(hash_function)

hashed_userid_rdd = user_idx_rdd.map(lambda x: (x[1], [f(x[1]) for f in hash_function_list]))
user_business_rdd = raw.map(lambda x: (user_idx_dict[x.split(',')[0]], business_idx_dict[x.split(',')[1]]))\
    .groupByKey().mapValues(list)
minhash_signature = user_business_rdd.leftOuterJoin(hashed_userid_rdd)\
    .flatMap(lambda record : [(business_idx, record[1][1]) for business_idx in record[1][0]])\
    .reduceByKey(lambda x, y: [min(i, j) for i, j in zip(x, y)])
candidate_pairs = set(minhash_signature.flatMap(lambda hashed_user: [(band, hashed_user[0]) for band in get_bands(hashed_user[1], 10)]).groupByKey().mapValues(list)
                      .map(lambda x: x[1]).filter(lambda candidates_list: len(candidates_list) > 1)
                      .flatMap(lambda candidates_list: [pair for pair in combinations(candidates_list, 2)]).collect())
business_user_dict = raw.map(lambda x: (business_idx_dict[x.split(',')[1]], user_idx_dict[x.split(',')[0]]))\
    .groupByKey().mapValues(list).collectAsMap()
idx_business_dict = {idx: business for business, idx in business_idx_dict.items()}

similar_pairs = []
for pair in candidate_pairs:
    jac_sim = jac(set(business_user_dict[pair[0]]), set(business_user_dict[pair[1]]))
    if jac_sim >= 0.5:
        pair = sorted(pair)
        similar_pairs.append([idx_business_dict[pair[0]], idx_business_dict[pair[1]], str(jac_sim)])

similar_pairs = sorted(similar_pairs)
with open(output_file_name, 'w+') as output:
    output.write('business_id_1,business_id_2,similarity\n')
    for items in similar_pairs:
        line = ""
        for item in items:
            line += ','+item
        line += '\n'
        output.write(line[1:])
    output.close()

end = time()
print(end-start)





