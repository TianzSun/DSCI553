from pyspark import SparkContext
from sklearn.metrics import mean_squared_error
import numpy as np


sc = SparkContext.getOrCreate()
output = sc.textFile("output.csv")\
    .filter(lambda x: x != "user_id,business_id,prediction").map(lambda x: x.split(',')).map(lambda x: float(x[2])).collect()
val = sc.textFile("yelp_val.csv")\
    .filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(',')).map(lambda x: float(x[2])).collect()

rmse = mean_squared_error(val, output)
lst = list(abs(np.array(val) - np.array(output)))
a = 0
b = 0
c = 0
d = 0
e = 0

for y in lst:
    if y < 1:
        a += 1
    elif 1 <= y < 2:
        b += 1
    elif 2 <= y < 3:
        c += 1
    elif 3 <= y < 4:
        d += 1
    else:
        e += 1


print(rmse**0.5)
print(a,b,c,d,e)

