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

ux0 = 1
uy0 = 0
# Creating the lattice with each point having nb_prob values associated with it
flow = np.array(nb_prob * width * height).reshape(height, width, len(nb_prob)) # initializing the lattice with probs

# list of directions possible for the point in the lattice
def lat_dir(ux, uy):
    return np.array([uy-ux, uy, ux+uy, -ux, 0, ux, -uy-ux, -uy, -uy+ux])

eu = lat_dir(ux0, 0)

#constructing the list for vectors of each direction possible at each point in the lattice

for e in range(9):  # e == 0-8 direction
    flow[:, :, e] *= nb_prob[e] * (1 + 3 * eu[e] + 4.5 * eu[e] ** 2 - 1.5 * ux0 ** 2)

flow_eq = np.ones((height, width, 9))


#----- initalizing macroscopic quantities

rho = np.ones((height, width))
ux = (flow[:, :, 1] + flow[:, :, 5] + flow[:, :, 8] - (flow[:, :, 3] + flow[:, :, 6] + flow[:, :, 7])) / rho
uy = (flow[:, :, 2] + flow[:, :, 5] + flow[:, :, 6] - (flow[:, :, 4] + flow[:, :, 7] + flow[:, :, 8])) / rho
u = np.sqrt(ux**2 + uy**2)

#------ initializing the border

domain = np.zeros((height, width), dtype=bool)
bot_wall = domain[0, :] = True
top_wall = domain[-1, :] = True
domain_out = np.zeros((height, width), dtype=bool)


#------- Sphere intialization

circle = {
            "centre": { "x": int(3*width/4), "y": int(height/2)},
            "radius": 20
          }


def domain_barrier(circle, domain):

    for x_cord in range(width):
        for y_cord in range(height):
            if np.sqrt((x_cord - circle["centre"]["x"])**2 + (y_cord - circle["centre"]["y"])**2) < circle["radius"]:
                domain[y_cord, x_cord] = True


domain_barrier(circle, domain)

def extract_boundary(domain):
    x_cord_list = []
    y_cord_list = []
    for x_cord in range(1, domain.shape[1]):
        for y_cord in range(1, domain.shape[0]):
            if domain[y_cord -1, x_cord]  and domain[y_cord, x_cord-1]:
                if domain[y_cord + 1, x_cord] and domain[y_cord, x_cord + 1]:
                    x_cord_list.append(x_cord)
                    y_cord_list.append(y_cord)
    domain[y_cord_list, x_cord_list] = False
    return domain
domain_boundary = extract_boundary(domain)
