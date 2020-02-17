import numpy as np

UNSIGNED_4 = np.uint8(4) # we want to do stuff modulo 4, if but if we compute np.uint64(3) % 4 numpy can't convince itself that a uint64 is big enough because the "4" is signed
