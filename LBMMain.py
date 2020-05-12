import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from numba import jit


class Domain:
    def __init__(self, height=100, width=200, circle = None):
        self.height = height
        self.width = width
        self.circle = circle

    @property
    def domain(self):
        return np.zeros((self.height, self.width), dtype=bool)

    def domain_topwall(self):
        d_top = self.domain
        d_top[-1, :] = True
        return d_top

    def domain_botwall(self):
        d_bot = self.domain
        d_bot[0, :] = True
        return d_bot

    def domain_barrier(self):
        _domain = self.domain
        for x_cord in range(self.width):
            for y_cord in range(self.height):
                if np.sqrt((x_cord - self.circle["centre"]["x"]) ** 2 + (y_cord - self.circle["centre"]["y"]) ** 2) <\
                        self.circle["radius"]:
                    _domain[y_cord, x_cord] = True
        return  _domain

    def domain_barrier_edge(self):
        _domain = self.domain_barrier()
        x_cord_list = []
        y_cord_list = []
        for x_cord in range(1, _domain.shape[1]):
            for y_cord in range(1, _domain.shape[0]):
                if _domain[y_cord - 1, x_cord] and _domain[y_cord, x_cord - 1]:
                    if _domain[y_cord + 1, x_cord] and _domain[y_cord, x_cord + 1]:
                        x_cord_list.append(x_cord)
                        y_cord_list.append(y_cord)

        _domain[y_cord_list, x_cord_list] = False
        return _domain


class LBM:

    def __init__(self, domain):
        self.domain = domain

        self.dx = 0.10  # distance step
        self.dt = 0.012  # timestep

        self.u0 = 0.18  # driven velocity
        self.l0 = 1.  # lattice step length
        self.v0 = 3e-6  # kinetic viscosity

        self.U0 = self.dt / self.dx  # characteristic velocity
        self.L0 = self.domain.width * self.dx  # characteristic length

        self.L_D = self.l0 / self.L0  # lattice length
        self.u_D = self.u0 / self.U0  # lattice velocity
        self.v_D = self.v0 / self.L0 / self.U0  # lattice viscosity

        self.Re = self.u_D * self.L_D / self.v_D  # Reynold's number

        self.cs = self.U0 / np.sqrt(3)  # lattice speed of sound
        self.tau = self.v0 / self.cs ** 2 + self.dt / 2.  # relaxation constant
        self.omega = self.dt / self.tau
        self.density_number = [1/36., 1/9., 1/36., 1/9., 4/9., 1/9., 1/36., 1/9., 1/36.]
        self.rho = np.zeros((height, width))
        self.f =  np.array(self.density_number * self.domain.width * self.domain.height).reshape(self.domain.height, self.domain.width, 9)
        self.f_eq = np.ones((height, width, 9))


    def rho_cal(self):  #check
        for j in range(9):
            self.rho[:, :] += self.f[:, :, j]

    def vel_cal(self):
        self.ux = (self.f[:,:,2] + self.f[:,:,5] + self.f[:,:,8] - (self.f[:,:,0] + self.f[:,:,3] + self.f[:,:,6])) / self.rho
        self.uy = (self.f[:,:,0] + self.f[:,:,1] + self.f[:,:,2] - (self.f[:,:,6] + self.f[:,:,7] + self.f[:,:,8])) / self.rho
        self.u = np.sqrt(self.ux**2 + self.uy**2)

    def pr_cal(self):
        outside = self.domain.domain_barrier_edge()
        self.px =  self.f[outside][:,0] * -self.ux[outside] + self.f[outside][:,3] * -self.ux[outside] + self.f[outside][:,6] * -self.ux[outside]
        self.py = self.f[outside][:, 0] * self.uy[outside] + self.f[outside][:, 6] * -self.uy[outside]

    def f_init(self):
        direction = self.lat_dir(self.u0, 0)
        self.f = self.lattice_vectors(direction, self.f, self.u0, self.density_number, callback="init")

    def lattice_vectors(self,direction, f, u, density_number, callback=None):
        if callback == "init":
            for e in range(9):  # e == 0-8 direction
                f[:, :, e] *= (1 + 3 * direction[e] + 4.5 * direction[e] ** 2 - 1.5 * u ** 2)
        else:
            for e in range(9):  # e == 0-8 direction
                f[:, :, e] = density_number * (1 + 3 * direction[e] + 4.5 * direction[e] ** 2 - 1.5 * u ** 2)

        return f

    def update_f(self):
        direction = self.lat_dir(self.ux, self.uy)
        self.lattice_vector_1(direction, self.f_eq, self.u)
        self.f = self.f + self.omega * (self.f_eq - self.f)

    def lattice_vector_1(self,direction, f, u):
        for e in range(9):  # e == 0-8 direction
            f[:, :, e] = self.rho * self.density_number[e] * (1 + 3 * direction[e] + 4.5 * direction[e] ** 2 - 1.5 * u ** 2)

    def flow_left(self):
        index_right= [2, 5, 8]
        index_left= [0, 3, 6]
        for index in index_left:
            self.f[:, -1, index] = self.density_number[index] * (1 + 3 * self.u0 - 1.5 * self.u0 ** 2 + 4.5 * self.u0 ** 2)
        for index in index_right:
            self.f[:, -1, index] = self.density_number[index] * (1 - 3 * self.u0 - 1.5 * self.u0 ** 2 + 4.5 * self.u0 ** 2)

    def streaming(self):
        f_copy = self.f.copy()
        for y in range(self.domain.height):
            for x in range(width):
                self.f[y, x] = self.stream(x, y, height, width, f_copy)

    def boundary(self):
        index_left = [0, 3, 6]
        index_right = [8, 5, 2]
        index_top = [6, 7, 8]
        index_bot = [0, 1, 2]
        f_copy = self.f.copy()
        #Bottom wall
        for index1, index2 in zip(index_top, index_bot):
            self.f[self.domain.domain_botwall(), index1] = self.f[self.domain.domain_botwall(), index2]

        # Top wall
        for index1, index2 in zip(index_bot, index_top):
            self.f[self.domain.domain_barrier_edge(), index1] = f_copy[self.domain.domain_barrier_edge(), index2]

        # Sphere wall
        for index1, index2 in zip(index_left, index_right):
            self.f[self.domain.domain_topwall(), index1] = self.f[self.domain.domain_topwall(), index2]

        # Sphere wall
        for index1, index2 in zip(index_right, index_left):
            self.f[self.domain.domain_topwall(), index1] = f_copy[self.domain.domain_topwall(), index2]

    @staticmethod
    def lat_dir(ux, uy):
        return np.array([uy - ux, uy, ux + uy, -ux, 0, ux, -uy - ux, -uy, -uy + ux])

    def stream(self, x, y, height, width, f_copy):
        return np.array([f_copy[min(y + 1, height - 1), x - 1, 0],
                         f_copy[min(y + 1, height - 1), x, 1],
                         f_copy[min(y + 1, height - 1), (x + 1) % width, 2],
                         f_copy[y, x - 1, 3],
                         f_copy[y, x, 4],
                         f_copy[y, (x + 1) % width, 5],
                         f_copy[max(y - 1, 0), x - 1, 6],
                         f_copy[max(y - 1, 0), x, 7],
                         f_copy[max(y - 1, 0), (x + 1) % width, 8]])

    def animation(self):
        X, Y = np.meshgrid(range(width), range(height))
        self.run()
        z = self.ux
        fig = plt.figure()
        ax = fig.gca()
        surf = ax.imshow(z, cmap='jet', interpolation='none')  # , vmin=-.18 , vmax=.4081)
        plt.xticks([])
        plt.yticks([])

        def update_data(i, z, surf):
            self.run()
            z = self.ux
            z[self.domain.domain_barrier()] = -.35
            ax.clear()
            ax.text(0, height - 1, ('frame {}'.format(i)))
            plt.xticks([])
            plt.yticks([])
            plt.title('$Re = {}$'.format(self.Re))
            surf = ax.imshow(z, cmap='jet', interpolation='none')  # , vmin=-.18 , vmax=.4081)
            return surf,
        anim = manimation.FuncAnimation(fig, update_data, fargs=(z, surf), interval=1, blit=False, repeat=True)
        plt.show()

    def run(self):
        self.f_init()
        self.rho_cal()# sets rho
        self.vel_cal()# sets vel
        self.pr_cal()# sets pressure
        self.update_f()# updates f
        self.flow_left()# check
        self.streaming()# check
        self.boundary()# apply boundary condition


if __name__ == "__main__":
    width = 200
    height = 100
    circle = {
        "centre": {"x": int(3 * width / 4), "y": int(height / 2)},
        "radius": 20
    }
    domain = Domain(width=width, height=height, circle=circle)

    lbm = LBM(domain)
    lbm.animation()



    print("debug")





# Equilibrium number density - Maxwell Boltzman distribution  - farther movements lesser probability
# 0 - rest(prob - 4/9 Highest probability) 1, 2, 3, 4 - slow (prob - 1/9 )  5, 6, 7, 8 - fast (prob - 1/36)
#
# # # probability weights for directions at a point in the lattice
# nb_prob = [1/36., 1/9., 1/36., 1/9., 4/9., 1/9., 1/36., 1/9., 1/36.]
# #
# height = 100 # lattice y length
# width = 200  # lattice x length
#
# ux0 = 1
# uy0 = 0
# dt = 0.012
# dx = 0.10
# u0 = 0.18   # driven velocity
# l0 = 1.     # lattice step length
# v0 = 3e-6
# U0 = dt / dx
# cs = U0 / np.sqrt(3)
# tau = v0 / cs**2 + dt/2.
# omega = dt / tau
# # Creating the lattice with each point having nb_prob values associated with it
# flow = np.array(nb_prob * width * height).reshape(height, width, len(nb_prob)) # initializing the lattice with probs
#
# # list of directions possible for the point in the lattice
# def lat_dir(ux, uy):
#     return np.array([uy-ux, uy, ux+uy, -ux, 0, ux, -uy-ux, -uy, -uy+ux])
#
# eu = lat_dir(ux0, 0)
#
# #constructing the list for vectors of each direction possible at each point in the lattice
#
# for e in range(9):  # e == 0-8 direction
#     flow[:, :, e] *= nb_prob[e] * (1 + 3 * eu[e] + 4.5 * eu[e] ** 2 - 1.5 * ux0 ** 2)
#
# flow_eq = np.ones((height, width, 9))
#
#
# #----- initalizing macroscopic quantities
#
# rho = np.ones((height, width))
# ux = (flow[:, :, 1] + flow[:, :, 5] + flow[:, :, 8] - (flow[:, :, 3] + flow[:, :, 6] + flow[:, :, 7])) / rho
# uy = (flow[:, :, 2] + flow[:, :, 5] + flow[:, :, 6] - (flow[:, :, 4] + flow[:, :, 7] + flow[:, :, 8])) / rho
# u = np.sqrt(ux**2 + uy**2)
#
# #------ initializing the border
#
# domain = np.zeros((height, width), dtype=bool)
# bot_wall = domain[0, :] = True
# top_wall = domain[-1, :] = True
# domain_out = np.zeros((height, width), dtype=bool)
#
#
# #------- Sphere intialization
#
# circle = {
#             "centre": { "x": int(3*width/4), "y": int(height/2)},
#             "radius": 20
#           }
#
#
# def domain_barrier(circle, domain):
#     for x_cord in range(width):
#         for y_cord in range(height):
#             if np.sqrt((x_cord - circle["centre"]["x"])**2 + (y_cord - circle["centre"]["y"])**2) < circle["radius"]:
#                 domain[y_cord, x_cord] = True
#
#
# domain_barrier(circle, domain)
#
# def extract_boundary(domain):
#     x_cord_list = []
#     y_cord_list = []
#     for x_cord in range(1, domain.shape[1]):
#         for y_cord in range(1, domain.shape[0]):
#             if domain[y_cord -1, x_cord]  and domain[y_cord, x_cord-1]:
#                 if domain[y_cord + 1, x_cord] and domain[y_cord, x_cord + 1]:
#                     x_cord_list.append(x_cord)
#                     y_cord_list.append(y_cord)
#     domain_copy = domain.copy()
#     domain_copy[y_cord_list, x_cord_list] = False
#     return domain_copy
# domain_boundary = extract_boundary(domain)
#
#
# @jit(nopython=True)
# def stream(x, y, height, width, f_copy)  :
#     return np.array([f_copy[min(y+1,height-1), x-1, 0],
#                     f_copy[min(y+1,height-1), x, 1],
#                     f_copy[min(y+1,height-1), (x+1)%width, 2],
#                     f_copy[y, x-1, 3],
#                     f_copy[y, x, 4],
#                     f_copy[y, (x+1)%width, 5],
#                     f_copy[max(y-1,0), x-1, 6],
#                     f_copy[max(y-1,0), x, 7],
#                     f_copy[max(y-1,0), (x+1)%width, 8]])
#
#
# global rho_tot, uxs, px_tot, py_tot
# rho_tot = []
# px_tot = []
# py_tot = []
#
# # global rho_tot, uxs, px_tot, py_tot
# # rho_tot = []
# # px_tot = []
# # py_tot = []
# def step() :
#     global flow, flow_eq
#     # ----- collision -----
#     rho = np.zeros((height, width))
#     for j in range(9) :
#         rho[:,:] += flow[:,:,j]  # eq.(6)
#     rho_tot.append(sum(rho.reshape(-1)))
#
#     ux = (flow[:,:,2] + flow[:,:,5] + flow[:,:,8] - (flow[:,:,0] + flow[:,:,3] + flow[:,:,6])) / rho
#     uy = (flow[:,:,0] + flow[:,:,1] + flow[:,:,2] - (flow[:,:,6] + flow[:,:,7] + flow[:,:,8])) / rho
#     u = np.sqrt(ux**2 + uy**2)
#     eu = np.array([uy-ux, uy, ux+uy, -ux, 0, ux, -uy-ux, -uy, -uy+ux]) # velocity vectors
#
#     px = flow[domain_boundary][:,0] * -ux[domain_boundary] + flow[domain_boundary][:,3] * -ux[domain_boundary] + flow[domain_boundary][:,6] * -ux[domain_boundary] # eq.(7)
#     py = flow[domain_boundary][:,0] * uy[domain_boundary] + flow[domain_boundary][:,6] * -uy[domain_boundary] # eq.(7)
#     px_tot.append(sum(px))
#     py_tot.append(sum(py))
#
#     for e in range(9) : # e == 0-8 direction
#         flow_eq[:,:,e] = rho * w[e] * (1 + 3*eu[e] + 4.5*eu[e]**2 - 1.5*u**2)  # eq.(5)
#     flow = flow + omega * (flow_eq - flow)
#
#     # flow to the left
#     f[:,-1,2] = w[2] * (1 + 3*u0 - 1.5*u0**2 + 4.5*u0**2) # eq.(5)
#     f[:,-1,5] = w[5] * (1 + 3*u0 - 1.5*u0**2 + 4.5*u0**2)
#     f[:,-1,8] = w[8] * (1 + 3*u0 - 1.5*u0**2 + 4.5*u0**2)
#     f[:,-1,0] = w[0] * (1 - 3*u0 - 1.5*u0**2 + 4.5*u0**2)
#     f[:,-1,3] = w[3] * (1 - 3*u0 - 1.5*u0**2 + 4.5*u0**2)
#     f[:,-1,6] = w[6] * (1 - 3*u0 - 1.5*u0**2 + 4.5*u0**2)
#     f_copy = f.copy()
#
#     # ----- streaming -----
#     for y in range(height) :
#         for x in range(width) :
#             f[y,x] = stream(x, y, height, width, f_copy)
#
#     f_copy = f.copy()
#     # walls
#     f[walltop, 6] = f[walltop, 0]
#     f[walltop, 7] = f[walltop, 1]      # bounceback top
#     f[walltop, 8] = f[walltop, 2]
#
#     f[wallbot, 2] = f_copy[wallbot, 8]
#     f[wallbot, 1] = f_copy[wallbot, 7]      # bounceback bottom
#     f[wallbot, 0] = f_copy[wallbot, 6]
#
#     f[barrier_outside, 3] = f[barrier_outside, 5]      # barrier object bounceback
#     f[barrier_outside, 0] = f[barrier_outside, 8]
#     f[barrier_outside, 6] = f[barrier_outside, 2]
#
#     f[barrier_outside, 2] = f_copy[barrier_outside, 6]
#     f[barrier_outside, 5] = f_copy[barrier_outside, 3]
#     f[barrier_outside, 8] = f_copy[barrier_outside, 0]
#
#     f[barrier_outside, 1] = f[barrier_outside, 7]
#     f[barrier_outside, 7] = f_copy[barrier_outside, 1]
#
#     return rho, ux, uy, u
#
# def animation() :
#     X,Y = np.meshgrid(range(width), range(height))
#     rho, ux, uy, u = step()
#     z = ux
#     fig = plt.figure()
#     ax = fig.gca()
#     surf = ax.imshow(z, cmap = 'jet', interpolation='none')#, vmin=-.18 , vmax=.4081)
#     plt.xticks([])
#     plt.yticks([])
#
#     def update_data(i, z, surf) :
#         rho, ux, uy, u = step()
#         z = ux
#         z[barrier] = -.35
#         ax.clear()
#         ax.text(0,height-1,('frame {}'.format(i)))
#         plt.xticks([])
#         plt.yticks([])
#         plt.title('$Re = {}$'.format(Re))
#         surf = ax.imshow(z, cmap = 'jet', interpolation='none')#, vmin=-.18 , vmax=.4081)
#         return surf,
#
#     anim = manimation.FuncAnimation(fig, update_data, fargs = (z,surf), interval = 1, blit = False, repeat = True)
#     plt.show()
#
# animation()
