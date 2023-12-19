"""
File to contain the Model class
"""


from matplotlib import pyplot as plt
import trimesh
import numpy as np
from typing import List

from geometry import Cut


class Model:
    """
    Class to generate a model from a series of faces
    """
    def __init__(self, dimension: List[float]) -> None:
        """
        Initialize a model made of a rectangle
        """
        self.dimension = dimension
        self.mesh = self.create_mesh()

    def create_mesh(self) -> trimesh.Trimesh:
        """
        Initialize 6 faces that form a box that represents an uncut model
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.dimension

        mesh = trimesh.creation.box(bounds=[[x_min, y_min, z_min],[x_max, y_max, z_max]])
        
        return mesh

    def plot(self) -> None:
        """
        plot 3D model mesh
        """
        ax = plt.gca()

        ax.plot_trisurf(self.mesh.vertices[:, 0],
                        self.mesh.vertices[:,1], 
                        triangles=self.mesh.faces, 
                        Z=self.mesh.vertices[:,2], 
                        alpha=.4)
    
    def make_cut(self, cut: Cut) -> None:
        """
        Using a defined cut, remove material from the model
        """
        self.mesh = trimesh.boolean.boolean_manifold([self.mesh, cut.mesh],
                                                     operation="difference")
        

def plot_mesh(mesh: trimesh.Trimesh) -> None:
    """
    Plot the mesh on an preexisting figure
    """
    ax = plt.gca()

    ax.plot_trisurf(mesh.vertices[:, 0],
                mesh.vertices[:,1], 
                triangles=mesh.faces, 
                Z=mesh.vertices[:,2], 
                alpha=.4)            
