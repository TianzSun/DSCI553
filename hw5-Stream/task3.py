import sys
from time import time
import random
from blackbox import BlackBox


start = time()

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

random.seed(553)
bb = BlackBox()
reservoir = []
lines = ["seqnum,0_id,20_id,40_id,60_id,80_id\n"]
n = 0
for i in range(num_of_asks):
    stream = bb.ask(input_filename, stream_size)
    if i == 0:
        for user in stream:
            reservoir.append(user)
        n = 100
    else:
        for user in stream:
            n += 1
            p = random.random()
            if p < 100/n:
                reservoir[random.randint(0, 99)] = user
    lines.append(str(n) + "," + reservoir[0] + "," + reservoir[20] + "," + reservoir[40] + "," + reservoir[60] + "," + reservoir[80] + "\n")

with open(output_filename, "w+") as out_file:
    for line in lines:
        out_file.writelines(line)

end = time()
print(end-start)
