"""
Main file to run the slicing algorithm and visualize the path planning
"""

import math
from matplotlib import pyplot as plt
import numpy as np
import itertools
from typing import Tuple
import vtkplotlib as vpl

from model import Model, STL_Model

class Slicing:
    """
    Class to generate the simplified version of the STL that can be cut
    """
    def __init__(self, path: str) -> None:
        """
        Initialize everything
        """
        self.stl_model = STL_Model(path)
        self.model = Model(self.stl_model.get_max_size())

    def generate_view_angles(self, nb_poses = 75) -> None:

        # visualize angles
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        r = 1
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

        view_angles = []

        for i in range(nb_poses):
            
            y = 1 - (i / float(nb_poses - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            if z > 0:

                # ax.scatter(x,y,z)
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')

                view_angles.append((x/math.sqrt(x**2 + y**2), math.acos(z/r)))

        return view_angles
    
    def rotate_and_flatten(self, perspective: Tuple[float]) -> np.ndarray:
        phi = -perspective[0]
        theta = perspective[1]


        array = self.stl_model.get_pointcloud()


        array = array @ np.array([[math.cos(theta), -math.sin(theta), 0],
                                  [math.sin(theta), math.cos(theta), 0],
                                  [0, 0, 1]])
        
        array = array @ np.array([[1, 0, 0],
                                  [0, math.cos(phi), -math.sin(phi)],
                                  [0, math.sin(phi), math.cos(phi)]])
        
        fig = plt.figure()


        array = array[:,0:3:2]
        plt.plot(array[:,0],array[:,1],"*")
        plt.gca().axis("equal")
        print(np.shape(array))
        
        return array
        


    def test(self):
        angles = self.generate_view_angles()
        shape = self.rotate_and_flatten(angles[0])




        


if __name__ == "__main__":
    slicing = Slicing("stl/dog.stl")

    

    slicing.test()
    # slicing.stl_model.plot_mesh()

    plt.show()
    