import numpy as np
from abstract import Barrier

class CircleBarrier(Barrier):

    def __init__(self, domain=None, radius=10, position=None, height=100, width=200):
        self.centre = (0, 0)
        self.radius = radius
        self.domain = domain # Add try except for domain argument
        self.height = height
        self.width = width
        self.position = position

    def apply_barrier(self): # lazy or not ??
        for x_cord in range(self.width):
            for y_cord in range(self.height):
                if np.sqrt((x_cord - self.position[0]) ** 2 + (y_cord - self.position[1]) ** 2) < \
                        self.radius:
                    self.domain[y_cord, x_cord] = True
        return self.domain

# class AirfoilBarrier(Barrier):
#
#     def __init__(self):
