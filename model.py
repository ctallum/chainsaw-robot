"""
File to contain the Model class
"""

from stl import mesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtkplotlib as vpl

import pymeshlab

import math
import numpy as np
from typing import List, Tuple


class Face:
    """
    Class to represent a single face of an object
    """
    def __init__(self, vertices: np.ndarray) -> None:
        self._vertices = vertices

    @property
    def vertices(self) -> np.ndarray:
        """
        Vertices property function of Face class 
        np.ndarray (2xn_points)
        """
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices) -> None:
        """
        Setter for vertices
        """
        self._vertices = vertices

    @property
    def area(self) -> float:
        """
        get signed area of face
        """
        area_sum = 0

        for idx,_ in enumerate(self.vertices[0] -1):
            x_1 = self.vertices[0, idx]
            y_1 = self.vertices[1, idx]
            x_2 = self.vertices[0, idx + 1]
            y_2 = self.vertices[1, idx + 1]

            area_sum += (x_2 - x_1)*(y_2 + y_1)

        x_1 = x_2
        x_2 = self.vertices[0,0]
        y_1 = y_2
        y_2 = self.vertices[1,0]
        area_sum += (x_2 - x_1)*(y_2 + y_1)

        return area_sum/2


class Model:
    """
    Class to generate a model from a series of faces
    """
    def __init__(self, dimension: List[float]) -> None:
        """
        Initialize a model made of a rectangle
        """

        self.dimension = dimension
        self.faces = self.initialize_faces()
        

    def initialize_faces(self) -> List[Face]:
        """
        Initialize 6 faces that form a box that represents an uncut model
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.dimension
        
        faces = []

        faces.append(Face(np.array([[x_min, x_max, x_max, x_min],
                                    [y_min, y_min, y_max, y_max],
                                    [z_min, z_min, z_min, z_min]])))
        
        faces.append(Face(np.array([[x_min, x_max, x_max, x_min],
                                    [y_min, y_min, y_max, y_max],
                                    [z_max, z_max, z_max, z_max]])))
        
        faces.append(Face(np.array([[x_min, x_min, x_min, x_min],
                                    [y_min, y_max, y_max, y_min],
                                    [z_min, z_min, z_max, z_max]])))
        
        faces.append(Face(np.array([[x_max, x_max, x_max, x_max],
                                    [y_min, y_max, y_max, y_min],
                                    [z_min, z_min, z_max, z_max]])))
        
        faces.append(Face(np.array([[x_min, x_max, x_max, x_min],
                                    [y_min, y_min, y_min, y_min],
                                    [z_min, z_min, z_max, z_max]])))
    
        faces.append(Face(np.array([[x_min, x_max, x_max, x_min],
                                    [y_max, y_max, y_max, y_max],
                                    [z_min, z_min, z_max, z_max]])))
        
        return faces
    
    def plot(self) -> None:
        """
        plot 3D model as a collection of faces
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.dimension

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        def drawpoly(xco, yco, zco):
            verts = [list(zip(xco, yco, zco))]
            temp = Poly3DCollection(verts)
            temp.set_facecolor("b")
            temp.set_alpha(.4)
            temp.set_edgecolor('k')

            ax.add_collection3d(temp)

        for face in self.faces:
            drawpoly(face.vertices[0], face.vertices[1], face.vertices[2])

        ax.set(xlim=(1.25*x_min, 1.25*x_max),
               ylim=(1.25*y_min, 1.25*y_max),
               zlim=(1.25*z_min, 1.25*z_max))
        
        ax.axis("equal")

    
class STL_Model:
    """
    class to hold information about the source STL model
    """
    def __init__(self,path: str) -> None:
        """
        Initialize STL model using input str path to model
        """
        self._path = path
        self.mesh = mesh.Mesh.from_file(self._path)
        self.pointcloud = self.get_pointcloud()

    def plot_mesh(self) -> None:
        """
        Plot the STL in a 3D plot
        """
        
        vpl.mesh_plot(self.mesh)
        vpl.show()
        

    def get_pointcloud(self) -> np.ndarray:
        """
        convert mesh into pointcloud
        """
        # cloud = open3d.geometry.sample_points_uniformly(input, number_of_points=samples)

        points = self.mesh.points.reshape([-1, 3])
        points = np.unique(points, axis=0)

        return points

    def plot_points(self) -> None:
        """
        Plot a scatter of the point cloud
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(self.pointcloud[:,0], self.pointcloud[:,1], self.pointcloud[:,2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    def get_max_size(self) -> Tuple[float]:
        """
        Get maximum size as bounding box for carving
        (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        points = self.pointcloud

        x_min = min(points[:,0])
        x_max = max(points[:,0])
        y_min = min(points[:,1])
        y_max = max(points[:,1])
        z_min = min(points[:,2])
        z_max = max(points[:,2])

        return (x_min, x_max, y_min, y_max, z_min, z_max)
