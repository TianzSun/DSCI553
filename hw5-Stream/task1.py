import sys
from time import time
import random
import binascii
from blackbox import BlackBox


def myhashs(s):
    result = []
    uid = int(binascii.hexlify(s.encode('utf8')), 16)
    for ab in ab_list:
        v = ((ab[0] * uid + ab[1]) % 114514) % 69997
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
bit_array = [0] * 69997
previous_users = set()
lines = ["Time,FPR\n"]

for n in range(num_of_asks):
    predict = [0] * stream_size
    stream = bb.ask(input_filename, stream_size)
    for i in range(stream_size):
        hash_val = myhashs(stream[i])
        bloom = 0
        for x in range(num_of_hashfun):
            if bit_array[hash_val[x]] == 1:
                bloom += 1
            else:
                break
        if bloom == num_of_hashfun:
            predict[i] = 1

    fp = 0
    for i in range(stream_size):
        if stream[i] not in previous_users and predict[i] == 1:
            fp += 1.0
    tn = len(predict)-sum(predict)
    fpr = fp / (fp + tn)
    lines.append(str(n) + "," + str(fpr) + "\n")

    for user in stream:
        hash_val = myhashs(user)
        for x in range(num_of_hashfun):
            bit_array[hash_val[x]] = 1
        previous_users.add(user)

with open(output_filename, "w+") as out_file:
    for line in lines:
        out_file.writelines(line)

end = time()
print(end-start)


