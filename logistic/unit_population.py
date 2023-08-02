'''
Unit test for computing one iteration of the logistic equation
'''
import numpy as np

N = 30 # number of steps to compute
x = 0.5
Lambda = 2.3

T = np.arange(N)
X = np.zeros(N)

for t in T:
    x = Lambda*x*(1-x)
    X[t] = x

import matplotlib.pyplot as plt

plt.plot(T, X, '-k')
plt.show()
