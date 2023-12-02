"""
Main file to run the slicing algorithm and visualize the path planning
"""

import math
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple
import vtkplotlib as vpl

from slicer import Slicer




        

if __name__ == "__main__":
    slicing = Slicer("stl/dog.stl")

    

    slicing.test()


    plt.show()
    plt.gca().axis("equal")