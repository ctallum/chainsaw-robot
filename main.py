"""
Main file to run the slicing algorithm and visualize the path planning
"""

import math
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple

from slicer import Slicer
from model import Model
from stl import STL_Model


path = "stl/dog.stl"

        

if __name__ == "__main__":
    # initialize everything
    stl = STL_Model(path)
    slicing = Slicer(path, num_poses = 10)
    model = Model(stl.get_max_size())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    stl.plot_points()

    for idx, cut in enumerate(slicing.cuts):
        model.make_cut(cut)


    print(idx)
    
    model.plot()
    # # stl.plot_mesh()
    # # model.mesh.show()

    plt.show()
