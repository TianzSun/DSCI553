import collections
import sys
from time import time
from itertools import combinations
from pyspark import SparkContext

filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

def SON_1(subset, support_num, count):
    if subset is None:
        return
    baskets_list = list(subset)
    scaled_down_sp = support_num * float(len(baskets_list) / count)
    single_item_dict = dict()
    final_dict = dict()
    for basket in baskets_list:
        for item in basket:
            if item not in single_item_dict.keys():
                single_item_dict[item] = 1
            else:
                single_item_dict[item] += 1
    filtered_dict = dict(filter(lambda x: x[1] >= scaled_down_sp, single_item_dict.items()))
    frequent_single_item_list = list(filtered_dict.keys())
    candidate_list = frequent_single_item_list
    c_len = 1
    while candidate_list is not None and len(candidate_list)>0:
        item_count_dict = dict()
        for basket in baskets_list:
            basket = list(set(basket).intersection(set(frequent_single_item_list)))
            for candidate in candidate_list:
                c_set = set()
                if type(candidate) == str:
                    c_set.add(candidate)
                else:
                    c_set = set(candidate)
                if c_set.issubset(set(basket)):
                    if candidate not in item_count_dict.keys():
                        item_count_dict[candidate] = 1
                    else:
                        item_count_dict[candidate] += 1
        filtered_dict = dict(filter(lambda x: x[1] >= scaled_down_sp, item_count_dict.items()))
        final_dict[c_len] = list(filtered_dict.keys())
        c_len += 1
        candidate_list = list()
        pre_list = sorted(list(filtered_dict.keys()))
        if len(pre_list) > 0:
            if type(pre_list[0]) == str:
                for pair in combinations(pre_list, 2):
                    candidate_list.append(pair)
            else:
                for i in range(len(pre_list) - 1):
                    base_tuple = pre_list[i]
                    for appender in pre_list[i + 1:]:
                        if base_tuple[:-1] == appender[:-1]:
                            new_tuple = tuple(sorted(list(set(base_tuple).union(set(appender)))))
                            candidate_list.append(new_tuple)
                        else:
                            break

    return final_dict.values()

def SON_2(subset,candidates):
    item_dict = dict()
    baskets_set = set()
    basket = list(subset)
    for ele in basket:
        baskets_set.add(ele)
    for c in candidates:
        c_set = set()
        if type(c) == str:
            c_set.add(c)
        else:
            c_set = set(c)
        if c_set.issubset(baskets_set):
            if c not in item_dict.keys():
                item_dict[c] = 1
            else:
                item_dict[c] += 1
    return item_dict.items()

sc = SparkContext.getOrCreate()
processed = sc.textFile(input_file_path).filter(lambda x: not x.startswith('"TRANSACTION_DT"'))\
    .map(lambda x : (x.split(',')[0][1:-1] + '-' + x.split(',')[1][1:-1], int(x.split(',')[5][1:-1]))).collect()

with open("customer_product.csv", 'w+') as output:
    output.write('DATE-CUSTOMER_ID,PRODUCT_ID\n')
    for line in processed:
        output.write(str(line[0])+','+str(line[1])+"\n")
output.close()

start = time()

baskets = sc.textFile("customer_product.csv").filter(lambda x: x != "DATE-CUSTOMER_ID,PRODUCT_ID")\
    .map(lambda x: (x.split(',')[0], x.split(',')[1])).groupByKey().mapValues(set)\
    .filter(lambda x: len(x[1]) > filter_threshold).map(lambda x: x[1])

basket_count = baskets.count()
candidates = baskets.mapPartitions(lambda x: SON_1(x, support, basket_count)).flatMap(lambda x: x)\
    .distinct().collect()
frequent = baskets.flatMap(lambda x: SON_2(x, candidates)).reduceByKey(lambda x, y: x+y)\
    .filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()

ans_1 = collections.defaultdict(list)
ans_2 = collections.defaultdict(list)

for x in candidates:
    if type(x) == str:
        x = "('" + x + "')"
        ans_1[0].append(x)
    else:
        x = sorted(x)
        ans_1[len(x) - 1].append("(" + str(x)[1:-1] + ")")

for x in frequent:
    if type(x) == str:
        x = "('" + x + "')"
        ans_2[0].append(x)
    else:
        x = sorted(x)
        ans_2[len(x) - 1].append("(" + str(x)[1:-1] + ")")

with open(output_file_path, 'w+') as output:
    output.write('Candidates:\n')
    for i in range(len(ans_1)):
        ans_1[i] = sorted(ans_1[i])
        line = ''
        for x in ans_1[i]:
            line += x + ','
        output.write(line[:-1]+'\n')
    output.write('Frequent Itemsets:\n')
    for i in range(len(ans_2)):
        ans_2[i] = sorted(ans_2[i])
        line = ''
        for x in ans_2[i]:
            line += x + ','
        output.write(line[:-1]+'\n')
output.close()

end = time()
print('Duration:', end - start)
