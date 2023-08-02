'''
Run the logistic eq and plot results
'''
import numpy as np
import matplotlib.pylab as plt

def logistic(r, x):
    '''
    The logistic equation
    '''
    return r * x * (1 - x)

N = 1000
T = np.arange(N)
last = 100
deltaLambda = 1000
x0 = 0.5
Lambdas = np.linspace(0.6, 3.5, deltaLambda)
x = x0 * np.ones_like(Lambdas)

fig = plt.figure(figsize=(8,6))

for t in T:
    x = logistic(Lambdas, x)
    
    #Display the plot
    if t >= (N - last):
        plt.plot(Lambdas, x, ',k', alpha=.25)
plt.show()