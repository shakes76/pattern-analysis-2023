'''
Unit test to test displaying random points
@author Shakes
'''
import numpy as np

n = 3

#generate n points along the unit circle
r = np.arange(0,n)
points = np.exp(2.0*np.pi*r*1j/n)
print("Points:", points)
