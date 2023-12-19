"""
Main file to run the slicing algorithm and visualize the path planning
"""

from slicer import Slicer
from model import Model
from stl import STL_Model
from geometry import Cut

import optimizer
from matplotlib import pyplot as plt
import time


path = "stl/dog.stl"

def is_not_done(slicing: Slicer):
    for cut in slicing.cuts:
        if not cut.is_cut:
            return True
    return False


if __name__ == "__main__":
    global_start_time = time.time()
    stl = STL_Model(path)
    num_poses = 10 # <--------------------------------------CHANGE THIS, the more num_poses, the more cuts and finer detail
    slicing = Slicer(path, num_poses)
    model = Model(stl.get_max_size())

    print("Number of poses: ",num_poses)

    surface_area = 0

    cut_order = []

    idx = 0

    while is_not_done(slicing):
        start_time = time.time()

        cut_idx = optimizer.give_next_cut_idx(model, slicing, "volume")
        # cut_idx = optimizer.give_next_cut_idx(model, slicing, "optimal") # <---------------- CHANGE THIS
        # cut_idx = optimizer.give_next_cut_idx(model, slicing, "random") # <----------------- CHANGE THIS

        if cut_idx == -1:
            break

        cut = slicing.cuts[cut_idx]
        cut_surface_area = cut.calculate_cut_surface_area(model.mesh)
        surface_area += cut_surface_area
        model.make_cut(cut)
        cut.is_cut = True

        cut_order.append(cut_idx)
        idx += 1
        duration = round((time.time() - start_time)*1000, 2)
        print("iteration: ",idx, ", cut index: ", cut_idx, ", cut surface: ",round(cut_surface_area,2), ", duration: ",duration, "ms")

    print("Cut Order: ",cut_order)
    print("Total Surface Area: ",surface_area)

    print("total time: ",round(1000*(time.time() - global_start_time),2),"ms")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    model.plot()
    stl.plot_points()

    plt.show()

