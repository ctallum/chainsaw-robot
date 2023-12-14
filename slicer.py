"""
File to contain all functions and classes related to the slicer, slices, and cuts
"""

from typing import List, Tuple
import math
from matplotlib import pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import os
import sys
import pandas as pd
from descartes import PolygonPatch
sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from model import STL_Model, Model


class Plane:
    """
    Class to contain methods needed to define an admissable plane that can be cut
    """
    def __init__(self, corners: np.ndarray)-> None:
        """
        Initialize a plane via the bounding corners
        """
        self.corners = corners

    def plot(self) -> None:
        """
        Given a 3D plot, plot the planes
        """
        def drawpoly(xco, yco, zco):
            verts = [list(zip(xco, yco, zco))]
            temp = Poly3DCollection(verts)
            temp.set_facecolor("b")
            temp.set_alpha(.05)
            temp.set_edgecolor('k')

            plt.gca().add_collection3d(temp)
        
        drawpoly(*self.corners.T)


class Cut:
    """
    Class to contain methods needed to define an admissable cut that removes material
    """
    def __init__(self, slices: List[Plane]):
        self.slices = slices
        self.mesh = self.generate_mesh()
        self.cost: float = None
        self.neighbors: List[Cut] = None
        
    def generate_mesh(self) -> None:
        return None

    def generate_cost(self) -> None:
        pass

    

class Slicer:
    """
    Class to contain methods needed to slice the object into a simplified 3D model
    """
    threshold = 0.1
    def __init__(self, path: str) -> None:
        """
        Initialize everything
        """
        self.stl_model = STL_Model(path)
        self.max_size = self.stl_model.get_max_size()
        self.model = Model(self.max_size)

        self.cuts: List[Cut] = []

        self.view_angles = self.generate_view_angles()

    @property
    def view_angles(self) -> List[Tuple[float,float]]:
        """
        View angles are a list of tuples containing the perspective in form [phi, theta]
        """
        return self.view_angles

    def generate_cuts(self) -> None:
        """
        Using the 3D model, generate a set of cuts which will remove material that bring the model
        to the correct shape
        """

        num_angles = len(self.view_angles)

        # for each possible angle, generate a set of admissable cuts
        for angle_idx in range(num_angles):
            angle = self.view_angles[angle_idx]
        
            # get a flattened version of the stl points from the specific angle
            perspective_array = self.rotate_and_flatten(angle)

            # get alpha shape which contains the set of points. Similar to a concave hull
            alpha_shape = alphashape.alphashape(perspective_array, 2)
            alpha_shape = alpha_shape.simplify(self.threshold, preserve_topology=True)
            alpha_shape = np.array(alpha_shape.boundary.xy)
            
            # weird issue where first and last points are the same
            if np.array_equal(alpha_shape[:,[0]], alpha_shape[:,[-1]]):
                alpha_shape = np.delete(alpha_shape, -1, 1)

            # if two sequential points are concave, remove points until we only have one concave
            # point. Prevents non-admissable cuts
            is_concave = self.get_concave_points(alpha_shape)
            alpha_shape, is_concave = self.remove_double_concave(alpha_shape, is_concave)

            # using the outline of the alpha shape, generate a set of 2d lines that extend outward
            # from the model
            cut_lines = self.generate_cut_lines(alpha_shape, is_concave)

            # start converting the 2D lines into 3D planes
            planes = []

            # take each 2D line and make it into a 3D plane and then rotate it into global frame
            for line_idx in range(len(cut_lines[0,0,:])):
                cut_line = cut_lines[:,:,line_idx]
                cut_points = np.zeros((3,4))
                cut_points[0,:] = np.concatenate((cut_line[0,:], np.flip(cut_line[0,:])), axis=0)
                cut_points[1,:] = np.array([1, 1, -1, -1])
                cut_points[2,: ] = np.concatenate((cut_line[1,:], np.flip(cut_line[1,:])), axis=0)
                cut_points_adjusted = self.unrotate(angle, cut_points.T)

                # make the plane into a Plane object by set of bounding coordinates
                planes.append(Plane(cut_points_adjusted))
            
            # add each plane as a cut. If the point is concave, add the set of two planes as a cut
            plane_idx = 0
            while plane_idx < len(planes) -1:
                if is_concave[plane_idx]:
                    self.cuts.append(Cut([planes[plane_idx], planes[plane_idx + 1]]))
                    plane_idx += 1
                else:
                    self.cuts.append(Cut([planes[plane_idx]]))
                plane_idx += 1


    def get_concave_points(self, points: np.ndarray) -> List[bool]:
        """
        Given a set of points that form the bounds of a polygon, return a list of bool which
        identify which points are the inner point of a concave feature.
        """

        # initialize empty bool list
        is_concave = []
        
        # for all points, check if the angle counterclockwise is greater than pi. If it is, append
        # true for concave
        for idx in range(len(points[0,:]) - 1):
            vertex_1 = points[:,[idx-1]]
            vertex_0 = points[:, [idx]]
            vertex_2 = points[:,[idx+1]]

            is_concave.append(self.angle(vertex_0, vertex_1, vertex_2) > math.pi)

        # check the last and first point
        vertex_1 = points[:,[-2]]
        vertex_0 = points[:,[-1]]
        vertex_2 = points[:,[0]]

        is_concave.append(self.angle(vertex_0, vertex_1, vertex_2) > math.pi)

        return is_concave

    def remove_double_concave(self, points: np.ndarray, is_concave: List[bool]
                              ) -> Tuple[np.ndarray, List[bool]]:
        """
        Given a set of points and a list of which ones are concave, remove points until there are
        no consecutive concave points
        """
        if is_concave[0] and is_concave[-1]:
            points = np.delete(points, 0, 1)
            is_concave.pop(0)
        
        cur_idx = 0
        while cur_idx < len(is_concave) - 2:
            if is_concave[cur_idx]:
                while is_concave[cur_idx + 1]:
                    points = np.delete(points, cur_idx + 1, 1)
                    is_concave.pop(cur_idx + 1)
            cur_idx += 1

        return points, is_concave
    

    def generate_cut_lines(self, points: np.ndarray, is_concave: bool) -> np.ndarray:
        """
        Given the set of points which define the alpha shape of a polygon, generate a set of 2D
        lines which represent cuts that follow the edge. 
        """
        cut_lines = np.zeros((2,2,len(points[0,:])))
        for idx in range(len(points[0,:])):
            prev_point = points[:,[idx-1]]
            cur_point = points[:,[idx]]

            if is_concave[idx-1]:
                vec = np.concatenate((prev_point, cur_point), axis=1)                
                vec = (vec - vec[:,[0]]) / np.linalg.norm(vec - vec[:,[0]]) * 2 + vec[:,[0]]
                
            elif is_concave[idx]:
                vec = np.concatenate((cur_point, prev_point), axis=1)
                vec =  (vec - vec[:,[0]]) / np.linalg.norm(vec - vec[:,[0]]) * 2 + vec[:,[0]]

            else:
                vec = np.concatenate((prev_point, cur_point), axis=1)
                original_len = np.linalg.norm(vec - vec[:,[0]])
                vec =  (vec - vec[:,[0]]) / original_len * 2 - .5*(2-original_len)*(vec[:,[1]]-vec[:,[0]])/original_len +  vec[:,[0]]

            cut_lines[:,:,idx] = vec

        return cut_lines


    def angle(self, vertex0: np.ndarray, vertex_1: np.ndarray, vertex_2: np.ndarray,
              angle_type: str ='unsigned') -> float:
        """
        Compute the angle between two edges  vertex0-- vertex_1 and  vertex0--
        vertex_2 having an endpoint in common. The angle is computed by starting
        from the edge  vertex0-- vertex_1, and then ``walking'' in a
        counterclockwise manner until the edge  vertex0-- vertex_2 is found.
        """
        # tolerance to check for coincident points
        tol = 2.22e-16

        # compute vectors corresponding to the two edges, and normalize
        vec1 = vertex_1 - vertex0
        vec2 = vertex_2 - vertex0

        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 < tol or norm_vec2 < tol:
            # vertex_1 or vertex_2 coincides with vertex0, abort
            edge_angle = math.nan
            return edge_angle

        vec1 = vec1 / norm_vec1
        vec2 = vec2 / norm_vec2

        # Transform vec1 and vec2 into flat 3-D vectors,
        # so that they can be used with np.inner and np.cross
        vec1flat = np.vstack([vec1, 0]).flatten()
        vec2flat = np.vstack([vec2, 0]).flatten()

        c_angle = np.inner(vec1flat, vec2flat)  # cos(theta) between two edges
        s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))

        edge_angle = math.atan2(s_angle, c_angle)

        angle_type = angle_type.lower()
        if angle_type == 'signed':
            # nothing to do
            pass
        elif angle_type == 'unsigned':
            edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
        else:
            raise ValueError('Invalid argument angle_type')

        return edge_angle
            

    def generate_view_angles(self, nb_poses = 10) -> List[Tuple[float, float]]:
        """
        Generate a set of perspectives to look at the model from, return list of [phi, theta]
        """
        nb_poses = nb_poses * 2

        view_angles = []

        r = 1
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(nb_poses):
            
            y = 1 - (i / float(nb_poses - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            if z > 0 :
                view_angles.append((math.atan2(y,x), math.acos(z/r)))
                
        return view_angles
    
    def rotate_and_flatten(self, perspective: Tuple[float]) -> np.ndarray:
        """
        Take a Tuple containing phi and theta, and from the perspective of those set of angles,
        flatten the point cloud. Return a 2D numpy array containing the new location of all the
        points
        """
        phi = -perspective[0]
        theta = perspective[1]

        array = self.stl_model.get_pointcloud()     

        array = array @ np.array([[math.cos(theta), -math.sin(theta), 0],
                                  [math.sin(theta), math.cos(theta), 0],
                                  [0, 0, 1]])
        
        array = array @ np.array([[1, 0, 0],
                                  [0, math.cos(phi), -math.sin(phi)],
                                  [0, math.sin(phi), math.cos(phi)]])
        
        return array[:,0:3:2]
    
    def unrotate(self, perspective: Tuple[float, float], cut_line: np.ndarray) -> np.ndarray:
        """
        Given a cut plane, unrotate it using the current perspective which contains phi and theta
        """
        phi = perspective[0]
        theta = -perspective[1]

        array = cut_line @ np.array([[1, 0, 0],
                                  [0, math.cos(phi), -math.sin(phi)],
                                  [0, math.sin(phi), math.cos(phi)]])
        
        array = array @ np.array([[math.cos(theta), -math.sin(theta), 0],
                                  [math.sin(theta), math.cos(theta), 0],
                                  [0, 0, 1]])
        return array
    

    def test(self):
        self.generate_cuts()
        self.stl_model.plot_points()
        for cut in self.cuts:
            for slice in cut.slices:
                slice.plot()
        self.cuts[0].slices[0].plot()

        print(sum([len(cut.slices) for cut in self.cuts]))


    