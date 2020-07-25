import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

I = np.random.choice(a = [0,1], size=5000)
A = np.zeros((5000,5000))
test = np.array([i for (i,j) in enumerate(I) if j ==1])
A[np.array([i for (i,j) in enumerate(I) if j ==1])] = A[:, np.array([i for (i,j) in enumerate(I) if j ==1])] = 0

print(test)