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

def compute_new_rand_location(startLoc):
    rand_location = np.random.randint(0, n)
    #print(rand_location, points[rand_location])
    vector = (points[rand_location] - startLoc)/2.0
    new_point = startLoc + vector
    return new_point, points[rand_location]

next_point, rand_location = compute_new_rand_location(start)

#plot
import matplotlib.pyplot as plt

#plot
# plt.plot(np.real(points[rand_location]), np.imag(points[rand_location]), "r.")
# plt.plot(np.real(start), np.imag(start), "g.")
# plt.plot(np.real(next_point), np.imag(next_point), "b.")
# plt.show()

iterations = 1000

next_point = start
for iteration in range(iterations):
    next_point, rand_location = compute_new_rand_location(next_point)
    plt.plot(np.real(next_point), np.imag(next_point), "b.")
    
#plot
plt.plot(np.real(points), np.imag(points), "r.")
plt.plot(np.real(start), np.imag(start), "g.")
plt.show()
