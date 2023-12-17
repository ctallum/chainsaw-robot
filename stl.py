"""
File to contain STL_Model class
"""
import trimesh
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple


class STL_Model:
    """
    class to hold information about the source STL model
    """
    def __init__(self,path: str) -> None:
        """
        Initialize STL model using input str path to model
        """
        self._path = path
        self.mesh = trimesh.load(self._path)
        self.pointcloud = self.get_pointcloud()

    def plot_mesh(self) -> None:
        """
        Plot the STL in a 3D plot
        """
        ax = plt.gca()

        ax.plot_trisurf(self.mesh.vertices[:, 0],
                        self.mesh.vertices[:,1], 
                        triangles=self.mesh.faces, 
                        Z=self.mesh.vertices[:,2], 
                        alpha=.4)
        
    def get_pointcloud(self) -> np.ndarray:
        """
        convert mesh into pointcloud
        """
        points = self.mesh.vertices.T
        points = np.unique(points, axis=1)

        return points

    def plot_points(self) -> None:
        """
        Plot a scatter of the point cloud
        """
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        ax = plt.gca()

        ax.scatter(self.pointcloud[0,:], self.pointcloud[1,:], self.pointcloud[2,:],"*k")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.axis("equal")

    def get_max_size(self) -> Tuple[float]:
        """
        Get maximum size as bounding box for carving
        (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        points = self.pointcloud


        x_min = min(points[0,:])
        x_max = max(points[0,:])
        y_min = min(points[1,:])
        y_max = max(points[1,:])
        z_min = min(points[2,:])
        z_max = max(points[2,:])

        return (x_min, x_max, y_min, y_max, z_min, z_max)