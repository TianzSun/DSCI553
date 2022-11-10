import sys
import json
from operator import add
from pyspark import SparkContext

review_filepath = sys.argv[1]
output_filepath = sys.argv[2]

ans = {}
sc = SparkContext.getOrCreate()

lines = sc.textFile(review_filepath).map(lambda x: json.loads(x))

# A. The total number of reviews

ans["n_review"] = lines.count()

# B. The number of reviews in 2018

ans["n_review_2018"] = lines.filter(lambda x: x["date"][:4] == "2018").count()

# C. The number of distinct users who wrote reviews

ans["n_user"] = lines.map(lambda x: x["user_id"]).distinct().count()

# D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote

ans["top10_user"] = lines.map(lambda x: (x["user_id"], 1)).reduceByKey(add).sortBy(lambda x: (-x[1], x[0])).take(10)

# E. The number of distinct businesses that have been reviewed

ans["n_business"] = lines.map(lambda x: x["business_id"]).distinct().count()

# F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had

ans["top10_business"] = lines.map(lambda x: (x["business_id"], 1)).reduceByKey(add).\
    sortBy(lambda x: (-x[1], x[0])).take(10)


with open(output_filepath, 'w+') as output:
    json.dump(ans, output)
output.close()

