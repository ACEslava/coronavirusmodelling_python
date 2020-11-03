import numpy as np
import random

I = np.array(np.zeros(10))
for x in random.sample(range(10), 2):
    I[x] = 1

print(I)