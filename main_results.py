"""
File to produce all the graphics in the report
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple

from slicer import Slicer
from model import Model
from stl import STL_Model
from geometry import Cut

import optimizer


import time


path = "stl/dog.stl"

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # initialize everything
    stl = STL_Model(path)
    stl.plot_points()
    ax.axis('equal')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    stl.plot_mesh()
    ax.axis('equal')


    num_poses = 10
    slicing = Slicer(path, num_poses, graphs=True)
    model = Model(stl.get_max_size())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    stl.plot_points()
    for cut_idx in range(28,28+6):
        cut = slicing.cuts[cut_idx]
        cut.plot()
        # model.make_cut(cut)
    model.plot()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    stl.plot_points()
    for cut_idx in range(28,28+6):
        cut = slicing.cuts[cut_idx]
        cut.plot()
        model.make_cut(cut)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    model.plot()
    stl.plot_points()
    plt.gca().axis('equal')

    for cut in slicing.cuts:
        model.make_cut(cut)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    model.plot()
    stl.plot_points()
    plt.gca().axis('equal')


    # stl.plot_mesh()
    # stl.plot_points()

    plt.show()