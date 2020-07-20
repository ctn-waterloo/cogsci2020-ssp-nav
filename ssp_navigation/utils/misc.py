import numpy as np


def gaussian_2d(x, y, meshgrid, sigma):
    X, Y = meshgrid
    return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / sigma / sigma)
