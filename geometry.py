"""
File to hold classes Plane and Cut
"""

import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Tuple

class Plane:
    """
    Class to contain methods needed to define an admissable plane that can be cut
    """
    def __init__(self, corners: np.ndarray) -> None:
        """
        Initialize a plane via the bounding corners
        """
        self.corners = corners
        self.point = self.corners[:,0]
        self.norm_vector = self.get_norm_vector()

    def get_norm_vector(self) -> np.ndarray:
        """
        Given the plane, find the normal vector pointing outward
        """
        vec_1 = self.corners[:,1] - self.corners[:,0]
        vec_2 = self.corners[:,2] - self.corners[:,1]

        # print(vec_1, vec_2)


        norm_vec = -(np.cross(vec_2, vec_1)) / np.linalg.norm(np.cross(vec_2, vec_1))

        # print(norm_vec)

        return norm_vec

        

    def plot(self) -> None:
        """
        Given a 3D plot, plot the planes
        """
        def drawpoly(xco, yco, zco):
            verts = [list(zip(xco, yco, zco))]
            temp = Poly3DCollection(verts)
            temp.set_facecolor("b")
            temp.set_alpha(.1)
            temp.set_edgecolor('k')

            plt.gca().add_collection3d(temp)
        
        drawpoly(*self.corners)

class Cut:

    """
    Class to contain methods needed to define an admissable cut that removes material
    """
    def __init__(self, slices: List[Plane]):
        self.slices = slices
        self.mesh = self.generate_mesh()
        self.cost: float = None
        self.neighbors: List[Cut] = None
        self.is_convex = len(self.slices) > 1

    def plot(self) -> None:
        """
        Plot plane or planes that define the cut
        """
        for slice in self.slices:
            slice.plot()
        
    def generate_mesh(self) -> None:
        return None

    def generate_cost(self) -> None:
        pass