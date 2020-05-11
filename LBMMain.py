import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from numba import jit

# Equilibrium number density - Maxwell Boltzman distribution  - farther movements lesser probability
# 0 - rest(prob - 4/9 Highest probability) 1, 2, 3, 4 - slow (prob - 1/9 )  5, 6, 7, 8 - fast (prob - 1/36)

# probability weights for directions at a point in the lattice
nb_prob = [1/36., 1/9., 1/36., 1/9., 4/9., 1/9., 1/36., 1/9., 1/36.]

height = 100 # lattice y length
width = 200  # lattice x length

# Creating the lattice with each point having nb_prob values associated with it
lattice = np.array(nb_prob * width * height).reshape(height, width, len(nb_prob)) # number densities

