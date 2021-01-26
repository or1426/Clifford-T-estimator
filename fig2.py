#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

qk_times = []
comp_times = []
FE_times = []
full_times = []
qs = list(range(26,51))
# for q in qs:
#     filename = "qk_calculate_comparison_{}.txt".format(q)
#     qk_time, comp_time = 0, 0
#     count = 0
#     with open(filename, "r") as f:
#         for line in f.readlines():
#             sline = line.split()
#             qk_time += float(sline[-2])
#             comp_time += float(sline[-1])
#             count += 1
#     qk_times.append(qk_time/count)
#     comp_times.append(comp_time/count)

for q in qs:
    filename = "no_qk_calculate_comparison2_{}.txt".format(q)
    comp_time = 0
    full_time = 0
    FE_time = 0
    count = 0
    with open(filename, "r") as f:
        for line in f.readlines():
            sline = line.split()
            #qk_time += float(sline[-2])
            full_time += float(sline[-3])
            FE_time += float(sline[-2])
            comp_time += float(sline[-1])
            count += 1
    #qk_times.append(qk_time/count)
    comp_times.append(comp_time/count)
    FE_times.append(FE_time/count)
    full_times.append(full_time/count)



#for q, comp_time in zip(qs, comp_times):
#    print(q, comp_time)
# plt.plot(qs, qk_times, label="qk")
plt.plot(qs, comp_times, label="compute")
plt.plot(qs, FE_times, label="feature extract")
plt.plot(qs, full_times, label="full run")

plt.legend()
plt.grid()
plt.xlabel("n")
plt.ylabel("mean time")
plt.yscale("log")

plt.title("t=30, depth=1000, w=10")
plt.show()

            
            
