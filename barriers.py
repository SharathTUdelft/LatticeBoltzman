import numpy as np
from abstract import Barrier

class CircleBarrier(Barrier):

    def __init__(self, domain=None, radius=10, position=(0, 0), height=100, width=200):
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

class AirfoilBarrier(Barrier):

    def __init__(self, max_camber=0.01, max_camber_loc = 0, ):
        pass

class BoxBarrier(Barrier):

    def __init__(self, domain=None, position=(0, 0), box_height=10, box_width=20,  height=100, width=200):

        self.domain = domain
        self.position = position
        self.box_height = box_height
        self.box_width = box_width
        self.height = height
        self.width = width

    def apply_barrier(self):
        for x_cord in range(self.width):
            for y_cord in range(self.height):
                if np.abs(x_cord - self.position[0]) < self.box_width and np.abs(y_cord - self.position[1]) < self.box_height:
                    self.domain[y_cord, x_cord] = True
        return self.domain
