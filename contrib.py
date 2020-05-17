import numpy as np
from abstract import Domain
from utils import lazy, Singleton
from barriers import CircleBarrier, BoxBarrier


class DomainCircle(Domain):

    def __init__(self, height, width):
        Domain.__init__(self, height=100, width=200)
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
        position = (int(3 * self.width / 4), int(self.height / 2))
        return CircleBarrier(domain=self.domain().copy(), position=position, radius=20, height =self.height, width = self.width ).apply_barrier()

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


class DomainBox(Domain):

    def __init__(self, height, width, box_height=10, box_width= 20):
        Domain.__init__(self, height=100, width=200)
        self.box_height = box_height
        self.box_width = box_width
        self.height = height
        self.width = width

    @lazy
    def domain_barrier(self):
        position = (int(3 * self.width / 4), int(self.height - 2 * self.box_height))
        return BoxBarrier(domain=self.domain().copy(), position=position, box_height=  self.box_height,box_width=self.box_width,  height =self.height, width = self.width ).apply_barrier()




# Tests
if __name__ == "__main__":
    domain = Domain()
    a1 = domain.domain()

    a2 = domain.domain_topwall()
    a3 = domain.domain_botwall()

    a4 = domain.domain_barrier()
    a5 = domain.domain_barrier_edge()

print("debug")