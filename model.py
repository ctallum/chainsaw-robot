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
                        alpha=.7)
    
    def make_cut(self, cut: Cut, debug=False) -> None:
        """
        Using a defined cut, remove material from the model
        """

        if not cut.is_convex:
            # get vertex information from the plane slice
            slice = cut.slices[0]
            vert_1 = slice.corners

            # create a new set of vertices that are parrallel offset from the original plane 
            vert_2 = vert_1 - np.reshape(slice.norm_vector,(3,1)) * 2*np.ones((1,4))

            # create a rectangular mesh using the original vertices and the offset vertices
            new_vertices = np.concatenate((vert_1.T, vert_2.T))
            new_faces = np.array([[0,1,2], [0,2,3], [4,1,0], [1,4,5],
                                  [6,5,4], [7,6,4], [6,7,2], [3,2,7],
                                  [5,2,1], [2,5,6], [0,3,4], [3,7,4]])
            square_mesh = trimesh.Trimesh(new_vertices, new_faces)
            
            # use the new rectangular mesh to remove material from the model mesh
            new_mesh = trimesh.boolean.boolean_manifold([self.mesh, square_mesh],
                                                        operation="difference")
            self.mesh = new_mesh

        else:
            # get the vertices from the two slices that form a wedge shape
            slice_1 = cut.slices[0]
            slice_2 = cut.slices[1]
            corners_1 = slice_1.corners
            corners_2 = slice_2.corners

            # create a new triangle prism mesh that connects the two planes
            new_vertices = np.concatenate((corners_1.T,corners_2.T))
            new_faces = np.array([[4,1,0], [1,2,3], [0,1,3], [2,1,4],
                                  [2,4,7], [6,7,4], [6,4,0], [7,3,2]])
            triangle_mesh = trimesh.Trimesh(new_vertices, new_faces)

            # use the new triangular mesh to remove material from the model mesh
            new_mesh = trimesh.boolean.boolean_manifold([self.mesh, triangle_mesh],
                                                        operation="difference")
            self.mesh = new_mesh

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
