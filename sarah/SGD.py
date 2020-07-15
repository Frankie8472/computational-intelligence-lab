import numpy as np

gamma = 0.005
lam = 0.02
k = 50

U = np.random.uniform(0, 0.05, (10000, k))
V = np.random.uniform(0, 0.05, (1000, k))

biasU = np.zeros(10000)
biasV = np.zeros(1000)

