'''
Unit test for random points
'''
import numpy as np

n = 3

#generate n points along the unit circle
r = np.arange(0,n)
points = np.exp(2.0*np.pi*r*1j/n)
print("Points:", points)

#coordinates of the unit circle
res = 100
w = np.arange(0,res)
unit_circle = np.exp(2.0*np.pi*w*1j/res)

#pick start position
start = np.random.randint(0, n)

#plot
import matplotlib.pyplot as plt

plt.plot(np.real(unit_circle), np.imag(unit_circle), "b-")
plt.plot(np.real(points), np.imag(points), "r.")
# plt.plot(np.real(start), np.imag(start), "g.")

plt.show()
