__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt

def show_2d_array(data, title=False):
    if title:
        plt.title(title)
    plt.imshow(data, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()
    plt.show()