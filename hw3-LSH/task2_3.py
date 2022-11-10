import sys
import math
from time import time
import json
import xgboost
import numpy as np
from pyspark import SparkContext


def get_average(score_list):
    sum = 0.0
    for x in score_list:
        sum += float(x[1])
    return sum / len(score_list)


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
        return [business_to_predict, business_avg_score.get(business_to_predict), len(neighbors_score_list)]
    return [business_to_predict, numerator / denominator, len(neighbors_score_list)]


def collaborative_filtering(pre_train_data, pre_test_data):
    user_idx_dict = pre_train_data.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
    business_idx_dict = pre_train_data.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
    idx_user_dict = {idx: user for user, idx in user_idx_dict.items()}
    idx_business_dict = {idx: business for business, idx in business_idx_dict.items()}

    business_user_rdd = pre_train_data.map(lambda x: (business_idx_dict[x[1]], (user_idx_dict[x[0]], x[2]))) \
        .groupByKey().mapValues(list)
    user_business_score_rdd = pre_train_data.map(lambda x: (user_idx_dict[x[0]], (business_idx_dict[x[1]], x[2]))) \
        .groupByKey().mapValues(list)

    business_user_score = business_user_rdd.collectAsMap()
    user_business_score = user_business_score_rdd.collectAsMap()
    business_avg_score = business_user_rdd.map(lambda x: (x[0], get_average(x[1]))).collectAsMap()
    user_avg_score = user_business_score_rdd.map(lambda x: (x[0], get_average(x[1]))).collectAsMap()

    test_user_business_rdd = pre_test_data.map(lambda x: (user_idx_dict.get(x[0], -1), business_idx_dict.get(x[1], -1))) \
        .filter(lambda x: x[0] != -1 and x[1] != -1)

    filtered_pairs = pre_test_data.filter(
        lambda pair: pair[0] not in user_idx_dict.keys() or pair[1] not in business_idx_dict.keys()).collect()

    joined_rdd = test_user_business_rdd.leftOuterJoin(user_business_score_rdd)
    candidate_pairs = joined_rdd.flatMap(lambda x: [(bus_score[0], x[1][0]) for bus_score in x[1][1]])

    business_pairs_dict = candidate_pairs \
        .filter(lambda pair: len(
        set(dict(business_user_score.get(pair[0])).keys()) & set(dict(business_user_score.get(pair[1])).keys())) >= 200) \
        .map(lambda pair: (
    pair, get_pearson(dict(business_user_score.get(pair[0])), dict(business_user_score.get(pair[1]))))) \
        .filter(lambda pair: pair[1] > 0).map(lambda pair: {(pair[0][0], pair[0][1]): pair[1]}) \
        .flatMap(lambda pair: pair.items()).collectAsMap()

    predict_res = joined_rdd.map(lambda x: (x[0], predict(x[1], business_pairs_dict, business_avg_score)))
    final_res = predict_res.map(
        lambda pair: ((idx_user_dict[pair[0]], idx_business_dict[pair[1][0]]), pair[1][1])).collect()
    predict_val_neighbor_num = predict_res.map(
        lambda pair: ((idx_user_dict[pair[0]], idx_business_dict[pair[1][0]]), pair[1][2])).collectAsMap()
    for pair in filtered_pairs:
        if pair[0] in user_idx_dict.keys():
            final_res.append((tuple(pair), user_avg_score[user_idx_dict[pair[0]]]))
            predict_val_neighbor_num[tuple(pair)] = len(user_business_score[user_idx_dict[pair[0]]])
        elif pair[1] in business_idx_dict.keys():
            final_res.append((tuple(pair), business_avg_score[business_idx_dict[pair[0]]]))
            predict_val_neighbor_num[tuple(pair)] = 0
        else:
            final_res.append((tuple(pair), 2.5))
            predict_val_neighbor_num[tuple(pair)] = 0
    return final_res, predict_val_neighbor_num


def model_based(pre_train_data, pre_test_data, user_feature_file, business_feature_file):
    user_to_train = set(pre_train_data.map(lambda x: x[0]).distinct().collect())
    business_to_train = set(pre_train_data.map(lambda x: x[1]).distinct().collect())

    user_feature_map = sc.textFile(user_feature_file).map(lambda x: json.loads(x)) \
        .filter(lambda x: x['user_id'] in user_to_train) \
        .map(lambda x: (x['user_id'], [x['review_count'], x['average_stars']])) \
        .collectAsMap()

    business_feature_map = sc.textFile(business_feature_file).map(lambda x: json.loads(x)) \
        .filter(lambda x: x['business_id'] in business_to_train) \
        .map(lambda x: (x['business_id'], [x['review_count'], x['stars']])) \
        .collectAsMap()

    train_x = np.array(pre_train_data.map(
        lambda x: np.array([user_feature_map[x[0]], business_feature_map[x[1]]]).flatten()).collect())
    train_y = np.array(pre_train_data.map(lambda x: float(x[2])).collect())
    test_x = np.array(pre_test_data.map(lambda x: np.array(
        [user_feature_map.get(x[0], [0, 2.5]), business_feature_map.get(x[1], [0, 2.5])]).flatten()).collect())

    model = xgboost.XGBRegressor(max_depth=10,
                                 learning_rate=0.1,
                                 n_estimators=100,
                                 objective='reg:linear',
                                 booster='gbtree',
                                 gamma=0,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 reg_alpha=0,
                                 reg_lambda=1,
                                 random_state=0)

    model.fit(train_x, train_y)
    predicted_value = model.predict(test_x)

    res = []
    for pair in zip(pre_test_data.collect(), predicted_value):
        res.append(((pair[0][0], pair[0][1]), pair[1]))
    return res


start = time()

sc = SparkContext.getOrCreate()

folder_path = sys.argv[1]+'/'
user_feature_file = folder_path + 'user.json'
business_feature_file = folder_path + 'business.json'
train_file_name = folder_path + 'yelp_train.csv'
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

pre_train_data = sc.textFile(train_file_name).filter(lambda x: x != "user_id,business_id,stars").map(
    lambda x: x.split(','))
pre_test_data = sc.textFile(test_file_name).filter(lambda x: x != "user_id,business_id,stars").map(
    lambda x: x.split(','))

mb_res = model_based(pre_train_data, pre_test_data, user_feature_file, business_feature_file)
cf_res, test_X_neighbor_num = collaborative_filtering(pre_train_data, pre_test_data)
max_neighbor_num = max(test_X_neighbor_num.values())
cf_normalized = []
mb_normalized = []
for pair in cf_res:
    cf_normalized.append((pair[0], float(test_X_neighbor_num[pair[0]] / max_neighbor_num) * pair[1]))
for pair in mb_res:
    mb_normalized.append((pair[0], (1 - float(test_X_neighbor_num[pair[0]] / max_neighbor_num)) * pair[1]))

combined_res = mb_normalized + cf_normalized
combined_rdd = sc.parallelize(combined_res).reduceByKey(lambda x, y: x + y)

with open(output_file_name, 'w+') as output:
    output.write('user_id,business_id,prediction\n')
    for pair in combined_rdd.collect():
        output.write(pair[0][0] + "," + pair[0][1] + "," + str(pair[1]) + "\n")
    output.close()

end = time()
print(end - start)
