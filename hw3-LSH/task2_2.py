import sys
import json
from time import time
import numpy as np
import xgboost
from pyspark import SparkContext

start = time()
sc = SparkContext.getOrCreate()

folder_path = sys.argv[1]+'/'
user_feature_file = folder_path + 'user.json'
business_feature_file = folder_path + 'business.json'
train_file_name = folder_path + 'yelp_train.csv'
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

pre_train_data = sc.textFile(train_file_name)\
    .filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))

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

train_x = np.array(pre_train_data.map(lambda x: np.array([user_feature_map[x[0]], business_feature_map[x[1]]]).flatten()).collect())
train_y = np.array(pre_train_data.map(lambda x: float(x[2])).collect())
test_x = np.array(sc.textFile(test_file_name).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))\
    .map(lambda x: np.array([user_feature_map.get(x[0], [0, 2.5]),business_feature_map.get(x[1], [0, 2.5])]).flatten()).collect())
test_rdd = sc.textFile(test_file_name).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))

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

res_rdd = test_rdd.map(lambda x: (x[0], x[1]))

with open(output_file_name, 'w+') as output:
    output.write('user_id,business_id,prediction\n')
    for pair in zip(res_rdd.collect(), predicted_value):
        output.write(pair[0][0] + "," + pair[0][1] + "," + str(pair[1]) + "\n")
    output.close()

end = time()
print(end - start)
