import sys
import math
from time import time
from pyspark import SparkContext


def get_average(score_list):
    sum = 0.0
    for x in score_list:
        sum += float(x[1])
    return sum/len(score_list)

def get_pearson(business1_scores, business2_scores):
    co_rated_users = list(set(business1_scores.keys()) & set(business2_scores.keys()))
    list1 = []
    list2 = []
    for user in co_rated_users:
        list1.append(float(business1_scores[user]))
        list2.append(float(business2_scores[user]))
    average1 = sum(list1) / len(list1)
    average2 = sum(list2) / len(list2)
    numerator = 0.0
    square_sum1 = 0.0
    square_sum2 = 0.0
    for score1, score2 in zip(list1, list2):
        numerator += ((score1 - average1) * (score2 - average2))
        square_sum1 += ((score1 - average1) * (score1 - average1))
        square_sum2 += ((score2 - average2) * (score2 - average2))
    if square_sum1 * square_sum2 == 0:
        return 0
    return numerator / (math.sqrt(square_sum1) * math.sqrt(square_sum2))


def predict(test_train_score, business_pairs_dict, business_avg_score):
    business_to_predict = test_train_score[0]
    neighbors_score_list = list(test_train_score[1])
    score_weight_list = []
    for business_score in neighbors_score_list:
        key = (business_score[0], business_to_predict)
        score_weight_list.append((float(business_score[1]), business_pairs_dict.get(key, 0)))
    top_50_score_list = sorted(score_weight_list, key=lambda score_weight: score_weight[1], reverse=True)[:50]

    numerator = 0.0
    denominator = 0.0
    for score_weight in top_50_score_list:
        numerator += (score_weight[0] * score_weight[1])
        denominator += abs(score_weight[1])
    if denominator * numerator == 0:
        return (business_to_predict, business_avg_score.get(business_to_predict))
    return (business_to_predict, numerator / denominator)


start = time()
sc = SparkContext.getOrCreate()

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]
train_rdd = sc.textFile(train_file_name).filter(lambda x: x != "user_id,business_id,stars")
test_rdd = sc.textFile(test_file_name).filter(lambda x: x != "user_id,business_id,stars")
user_idx_dict = train_rdd.map(lambda x: x.split(',')[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
business_idx_dict = train_rdd.map(lambda x: x.split(',')[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
idx_user_dict = {idx: user for user, idx in user_idx_dict.items()}
idx_business_dict = {idx: business for business, idx in business_idx_dict.items()}

train_business_user_rdd = train_rdd.map(lambda x: (business_idx_dict[x.split(',')[1]], (user_idx_dict[x.split(',')[0]], x.split(',')[2])))\
    .groupByKey().mapValues(list)
business_user_score = train_business_user_rdd.collectAsMap()

business_avg_score = train_business_user_rdd.map(lambda record: (record[0], get_average(record[1]))).collectAsMap()

test_user_business_rdd = test_rdd.map(lambda x: (user_idx_dict.get(x.split(',')[0], -1), business_idx_dict.get(x.split(',')[1], -1)))\
    .filter(lambda record: record[0] != -1 and record[1] != -1)

filtered_pairs = test_rdd.map(lambda x: x.split(','))\
    .filter(lambda x: x[0] not in user_idx_dict or x[1] not in business_idx_dict).collect()

user_business_score_rdd = train_rdd.map(lambda x: (user_idx_dict[x.split(',')[0]], (business_idx_dict[x.split(',')[1]], x.split(',')[2])))\
    .groupByKey().mapValues(list)

user_business_score = user_business_score_rdd.collectAsMap()
user_avg_score = user_business_score_rdd.map(lambda record: (record[0], get_average(record[1]))).collectAsMap()

joined_rdd = test_user_business_rdd.leftOuterJoin(user_business_score_rdd)
candidate_pairs = joined_rdd.flatMap(lambda record: [(bus_score[0], record[1][0]) for bus_score in record[1][1]])

business_pairs_dict = candidate_pairs\
    .filter(lambda x: len(set(dict(business_user_score.get(x[0])).keys()) & set(dict(business_user_score.get(x[1])).keys())) >= 300)\
    .map(lambda x: (x, get_pearson(dict(business_user_score.get(x[0])), dict(business_user_score.get(x[1])))))\
    .filter(lambda x: x[1] > 0).map(lambda x: {(x[0][0], x[0][1]): x[1]})\
    .flatMap(lambda x: x.items()).collectAsMap()

final_res_rdd = joined_rdd.map(lambda record: (record[0], predict(record[1], business_pairs_dict, business_avg_score)))

with open(output_file_name, 'w+') as output:
    output.write('user_id,business_id,prediction\n')
    for pair in final_res_rdd.collect():
        output.write(idx_user_dict[pair[0]]+","+idx_business_dict[pair[1][0]]+","+str(pair[1][1])+"\n")
    for pair in filtered_pairs:
        if pair[0] in user_idx_dict.keys():
            output.write(pair[0] + "," + pair[1] + "," + str(user_avg_score[user_idx_dict[pair[0]]]) + "\n")
        elif pair[1] in business_idx_dict.keys():
            output.write(pair[0] + "," + pair[1] + "," + str(business_avg_score[business_idx_dict[pair[0]]]) + "\n")
        else:
            output.write(pair[0]+","+pair[1]+","+str(0.0)+"\n")
    output.close()

end = time()
print(end-start)

