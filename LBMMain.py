import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from numba import jit
from utils import lazy





class Domain:


    def __init__(self, height=100, width=200, circle = None):
        self.height = height
        self.width = width
        self.circle = circle

    @lazy
    def domain(self):
        return np.zeros((self.height, self.width), dtype=bool)

    @lazy
    def domain_botwall(self):
        d_top = self.domain().copy()
        d_top[-1, :] = True
        return d_top

    @lazy
    def domain_topwall(self):
        d_bot = self.domain().copy()
        d_bot[0, :] = True
        return d_bot

    @lazy
    def domain_barrier(self):
        _domain = self.domain().copy() # work with copy as self.domain is lazy itself
        for x_cord in range(self.width):
            for y_cord in range(self.height):
                if np.sqrt((x_cord - self.circle["centre"]["x"]) ** 2 + (y_cord - self.circle["centre"]["y"]) ** 2) <\
                        self.circle["radius"]:
                    _domain[y_cord, x_cord] = True
        return _domain


    @lazy
    def domain_barrier_edge(self):
        _domain = self.domain_barrier().copy() # work with copy as self.domain_barrier is lazy itself
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
        self.f =  np.array(self.density_number * self.domain.width * self.domain.height).reshape(self.domain.height, self.domain.width, 9)
        self.f_eq = np.ones((height, width, 9))

        self.f_init()
        self.rho_cal()  # sets rho
        self.vel_cal()  # sets vel


    def rho_cal(self):  #check
        self.rho = np.zeros((self.domain.height, self.domain.width))
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
        self.f = self.lattice_vectors(direction, self.f, self.u0, self.density_number)


    def lattice_vectors(self,direction, f, u):
        for e in range(9):  # e == 0-8 direction
            f[:, :, e] *= (1 + 3 * direction[e] + 4.5 * direction[e] ** 2 - 1.5 * u ** 2)
        return f


    def update_f(self):
        direction = self.lat_dir(self.ux, self.uy)
        self.lattice_vector_vel(direction, self.f_eq, self.u)
        self.f = self.f + self.omega * (self.f_eq - self.f)


    def lattice_vector_vel(self,direction, f, u):
        for e in range(9):  # e == 0-8 direction
            f[:, :, e] = self.rho * self.density_number[e] * (1 + 3 * direction[e] + 4.5 * direction[e] ** 2 - 1.5 * u ** 2)


    def flow_left(self):
        index_right= [2, 5, 8]
        index_left= [0, 3, 6]
        for index in index_right:
            self.f[:, -1, index] = self.density_number[index] * (1 + 3 * self.u0 - 1.5 * self.u0 ** 2 + 4.5 * self.u0 ** 2)
        for index in index_left:
            self.f[:, -1, index] = self.density_number[index] * (1 - 3 * self.u0 - 1.5 * self.u0 ** 2 + 4.5 * self.u0 ** 2)


    def streaming(self):
        f_copy = self.f.copy()
        for y in range(self.domain.height):
            for x in range(self.domain.width):
                self.f[y, x] = self.stream(x, y, height, width, f_copy)


    def boundary(self):
        index_left = [0, 3, 6]
        index_right = [8, 5, 2]
        index_top = [6, 7, 8]
        index_bot = [0, 1, 2]
        f_copy = self.f.copy()

        topwall = self.domain.domain_topwall()
        botwall = self.domain.domain_botwall()
        barrier_outside = self.domain.domain_barrier_edge()
        #Bottom wall
        for index1, index2 in zip(index_top, index_bot):
            self.f[botwall, index1] = self.f[botwall, index2]

        # Top wall
        for index1, index2 in zip(index_bot, index_top):
            self.f[topwall, index1] = f_copy[topwall, index2]

        # Sphere wall
        for index1, index2 in zip(index_left, index_right):
            self.f[barrier_outside, index1] = self.f[barrier_outside, index2]

        # Sphere wall
        for index1, index2 in zip(index_right, index_left):
            self.f[barrier_outside, index1] = f_copy[barrier_outside, index2]

        self.f[barrier_outside, 1] = self.f[barrier_outside, 7]
        self.f[barrier_outside, 7] = f_copy[barrier_outside, 1]

    @staticmethod
    def lat_dir(ux, uy):
        return np.array([uy - ux, uy, ux + uy, -ux, 0, ux, -uy - ux, -uy, -uy + ux])


    @staticmethod
    @jit(nopython=True)
    def stream(x, y, height, width, f_copy):
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
        X, Y = np.meshgrid(range(self.domain.width), range(self.domain.height))
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

        self.vel_cal()  # sets vel
        self.rho_cal()# sets rho
        self.vel_cal()  # sets vel
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
    # domain.domain_barrier()
    # domain.domain_barrier_edge()
    # domain.domain_barrier()

    print("debug")



