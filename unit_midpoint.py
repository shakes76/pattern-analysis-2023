'''
Unit test for random points
'''
import numpy as np

n = 3

#generate n points along the unit circle
r = np.arange(0,n)
points = np.exp(2.0*np.pi*r*1j/n)
print("Points:", points)

#pick start position
start = 0.1+0.5j

rand_location = np.random.randint(0, n)
print(rand_location, points[rand_location])
vector = (points[rand_location] - start)/2.0
new_point = start + vector

#plot
import matplotlib.pyplot as plt

#plot
plt.plot(np.real(points[rand_location]), np.imag(points[rand_location]), "r.")
plt.plot(np.real(start), np.imag(start), "g.")
plt.plot(np.real(new_point), np.imag(new_point), "b.")
plt.show()
