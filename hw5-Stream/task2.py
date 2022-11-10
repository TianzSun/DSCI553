import sys
from time import time
import random
import binascii
from blackbox import BlackBox


def myhashs(s):
    result = []
    uid = int(binascii.hexlify(s.encode('utf8')), 16)
    for ab in ab_list:
        v = ((ab[0] * uid + ab[1]) % 69997) % 1919
        result.append(v)
    return result


start = time()

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

num_of_hashfun = 7
ab_list = []
for i in range(num_of_hashfun):
    ab = [random.randint(1, sys.maxsize-1), random.randint(0, sys.maxsize-1)]
    ab_list.append(ab)

bb = BlackBox()
sum_gt = 0
sum_est = 0
lines = ["Time,Ground Truth,Estimation\n"]

for n in range(num_of_asks):
    stream = bb.ask(input_filename, stream_size)
    lst = [0] * num_of_hashfun
    for i in range(stream_size):
        hash_val = myhashs(stream[i])
        for j in range(num_of_hashfun):
            lst[j] = max(len(bin(hash_val[j]).split("1")[-1]), lst[j])

    gt = len(set(stream))
    est = 0
    for x in lst:
        est += 2 ** x
    est = round(est / num_of_hashfun)
    sum_gt += gt
    sum_est += est
    lines.append(str(n) + "," + str(gt) + "," + str(est) + "\n")

with open(output_filename, "w+") as out_file:
    for line in lines:
        out_file.writelines(line)
print(sum_est / sum_gt)
end = time()
print(end-start)
