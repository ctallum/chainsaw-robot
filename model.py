"""
File to contain the Model class
"""


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import vtkplotlib as vpl

import pymeshlab

import trimesh

import math
import numpy as np
from typing import List, Tuple

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
    
    def make_cut(self, cut: Cut, debug=False) -> None:
        """
        Using a defined cut, remove material from the model
        """

        # case 1: cut is non-convex
        if not cut.is_convex:
            # self.mesh = self.mesh.slice_plane(cut.slices[0].point, cut.slices[0].norm_vector, cap = True)
            # trimesh.repair.fill_holes(self.mesh)
            # trimesh.repair.fix_inversion(self.mesh)

            slice = cut.slices[0]

            vert_1 = slice.corners
            # print(vert_1.shape, slice.norm_vector.shape)
            vert_2 = vert_1 - np.reshape(slice.norm_vector,(3,1)) * 2*np.ones((1,4))

            new_vertices = np.concatenate((vert_1.T, vert_2.T))

            new_faces = np.array([[0,1,2],[0,2,3],[4,1,0], [1,4,5], [6,5,4],[7,6,4], [6,7,2], [3,2,7], [5,2,1], [2,5,6], [0,3,4],[3,7,4]])

            square_mesh = trimesh.Trimesh(new_vertices, new_faces)
            
            # square_mesh.show()


            # ax = plt.gca()

            # ax.plot_trisurf(square_mesh.vertices[:, 0],
            #             square_mesh.vertices[:,1], 
            #             triangles=square_mesh.faces, 
            #             Z=square_mesh.vertices[:,2], 
            #             alpha=.4)
            new_mesh = trimesh.boolean.boolean_manifold([self.mesh, square_mesh], operation="difference")
            self.mesh = new_mesh
            return



        else:   
            # print("doing concave cut")

            slice_1 = cut.slices[0]
            slice_2 = cut.slices[1]

            corners_1 = slice_1.corners
            corners_2 = slice_2.corners

            # print(corners_1)
            # print()
            # print(corners_2)
            

            v_1, v_2, v_3, v_4 = tuple(corners_1.T)
            v_5, v_6, v_7, v_8 = tuple(corners_2.T)


            new_vertices = np.concatenate((corners_1.T,corners_2.T))
            
            new_faces = np.array([[4, 1,0], [1,2,3], [0,1,3], [2,1,4],[2,4,7],[6,7,4],[6,4,0],[7,3,2]])

            triangle_mesh = trimesh.Trimesh(new_vertices, new_faces)

            if debug:
                print("is triangle volume: ",triangle_mesh.is_volume)

            # triangle_mesh.show()
                
                # triangle_mesh.show()

                # ax = plt.gca()

                # ax.plot_trisurf(triangle_mesh.vertices[:, 0],
                #             triangle_mesh.vertices[:,1], 
                #             triangles=triangle_mesh.faces, 
                #             Z=triangle_mesh.vertices[:,2], 
                #             alpha=.4)
                # return




            mesh_1 = self.mesh.slice_plane(cut.slices[0].point, cut.slices[0].norm_vector, cap = True)
            mesh_2 = self.mesh.slice_plane(cut.slices[1].point, cut.slices[1].norm_vector, cap = True)

            # new_mesh = self.join_two_meshes(mesh_1, mesh_2)

            # print(new_mesh.vertices)
            # trimesh.repair.fill_holes(new_mesh)
            # trimesh.repair.fix_inversion(new_mesh)
            # # new_mesh.remove_duplicate_faces()
            # new_mesh.remove_unreferenced_vertices()

            new_mesh = trimesh.boolean.boolean_manifold([self.mesh, triangle_mesh],operation="difference")

            

            self.mesh = new_mesh
            # self.mesh = mesh_1
            


            
    def join_two_meshes(self, mesh_1: trimesh.Trimesh, mesh_2: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Merge 2 meshes together using similar vertices and faces
        """




        new_mesh = trimesh.Trimesh.union(mesh_1, mesh_2, engine="manifold")

        



        return new_mesh




        



            
