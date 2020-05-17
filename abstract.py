import numpy as np
from utils import lazy, Singleton


# class AbstractDomain(Singleton): # NO need for abstract domain as it already is a singleton
#
#     def domain(self):
#         raise NotImplementedError
#
#     def domain_botwall(self):
#         raise NotImplementedError
#
#     def domain_topwall(self):
#         raise NotImplementedError
#
#     def domain_barrier(self):
#         raise NotImplementedError
#
#     def domain_barrier_edge(self):
#         raise NotImplementedError

class Domain(Singleton):

    def __init__(self, height=100, width=200):
        self.height = height
        self.width = width

    @lazy
    def domain(self):
        return np.zeros((self.height, self.width), dtype=bool)

    @lazy
    def domain_topwall(self):
        d_top = self.domain().copy()
        d_top[0, :] = True
        return d_top

    @lazy
    def domain_botwall(self):
        d_top = self.domain().copy()
        d_top[-1, :] = True
        return d_top

    @lazy
    def domain_barrier(self):
        pass

    @lazy
    def domain_barrier_edge(self):
        _domain = self.domain_barrier().copy()  # work with copy as self.domain_barrier is lazy itself
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


class Barrier:

    def apply_barrier(self):
        raise NotImplementedError

