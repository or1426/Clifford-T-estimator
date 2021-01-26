#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import pickle

from matplotlib import rc

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], "size": 22})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino'], "size":26})
rc('text', usetex=True)

# data = None
# with open("v_r_data.pkl", "rb") as f:
#     data = pickle.load(f)
    
# data = 40 - np.array(data)[:,1]

# m, M = min(data), max(data)

# freqs = [0 for _ in range(m, M+1)]
# for d in data:
#     freqs[d-m] += 1

# print(m, M)
# print(np.array(range(m,M+1)).shape)
# print(np.array(freqs).shape)


# plt.bar(np.array(range(m,M+1)), freqs, color='g', alpha=0.8, tick_label=np.array(range(m,M+1)))
# plt.grid()
# plt.xlabel("t-r")
# plt.ylabel("count")

# plt.savefig("../t-r-26.pdf",bbox_inches="tight")

#plt.show()


# log_vs = []
# rs = []

# with open("vr_info.txt", "r") as f:
#     for i, line in enumerate(f.readlines()):
        
#         stuff = line.split(",")
#         if len(stuff) == 2:
#             log_vs.append(int(stuff[0]))
#             rs.append(int(stuff[1]))


            
# plt.hist(log_vs)
# plt.legend()
# plt.grid()
# plt.xlabel("log_v")
# plt.ylabel("frequency")

# plt.figure()
# plt.hist(rs)
# plt.xlabel("r")
# plt.ylabel("frequency")


# plt.show()
