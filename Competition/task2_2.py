import sys
import json
from time import time
import numpy as np
import xgboost
from pyspark import SparkContext

"""
Method Description:
In this program, I use the Model-based method. 
For user, I choose review count and average stars as the feature,
and for business, I choose review count and stars as the feature.
Then use XGBregressor to train a model.

Error Distribution:
>=0 and <1: 101552
>=1 and <2: 33317
>=2 and <3: 6350
>=3 and <4: 824
>=4: 1

RMSE:
0.9867087206468333

Execution Time:
83s
"""

start = time()
sc = SparkContext.getOrCreate()
"""
folder_path = sys.argv[1]+'/'
user = folder_path + 'user.json'
business = folder_path + 'business.json'
train_file_name = folder_path + 'yelp_train.csv'
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]
"""

user_file = 'user.json'
business_file = 'business.json'
train_file_name = 'yelp_train.csv'
test_file_name = 'yelp_val.csv'
output_file_name = 'output.csv'

raw_data = sc.textFile(train_file_name)\
    .filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))

user = raw_data.map(lambda x: x[0]).distinct().collect()
business = raw_data.map(lambda x: x[1]).distinct().collect()
star = raw_data.map(lambda x: float(x[2])).collect()
default_star = sum(star) / len(star)

user_feature = sc.textFile(user_file).map(lambda x: json.loads(x)) \
    .filter(lambda x: x['user_id'] in user) \
    .map(lambda x: (x['user_id'], [x['review_count'], x['average_stars']])).collectAsMap()

business_feature = sc.textFile(business_file).map(lambda x: json.loads(x)) \
    .filter(lambda x: x['business_id'] in business) \
    .map(lambda x: (x['business_id'], [x['review_count'], x['stars']])).collectAsMap()

train_x = np.array(raw_data.map(lambda x: np.array([user_feature[x[0]], business_feature[x[1]]]).flatten()).collect())
train_y = np.array(star)
test_x = np.array(sc.textFile(test_file_name)
                  .filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(',')) \
                  .map(lambda x: np.array([user_feature.get(x[0], [0, default_star]), business_feature.get(x[1], [0, default_star])]).flatten()).collect())
test_data = sc.textFile(test_file_name).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))

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

prediction = model.predict(test_x)

ans = test_data.collect()
for i in range(len(ans)):
    ans[i][2] = str(prediction[i])

with open(output_file_name, 'w+') as output:
    output.write('user_id,business_id,prediction\n')
    for line in ans:
        output.write(line[0] + "," + line[1] + "," + line[2] + "\n")
    output.close()

end = time()
print(end - start)
