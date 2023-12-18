"""
File to contain all functions and classes related to the slicer, slices, and cuts
"""

from typing import List, Tuple
import math
from matplotlib import pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.spatial import ConvexHull


import alphashape

from stl import STL_Model
from geometry import Cut, Plane



class Slicer:
    """
    Class to contain methods needed to slice the object into a simplified 3D model
    """
    threshold = 0.1
    def __init__(self, path: str, num_poses: int = 20) -> None:
        """
        Initialize everything
        """
        self.stl_model = STL_Model(path)
        self.max_size = self.stl_model.get_max_size()
        self.cuts: List[Cut] = []
        self.view_angles = self.generate_view_angles(nb_poses = num_poses)
        self.generate_cuts()


    def generate_cuts(self) -> None:
        """
        Using the 3D model, generate a set of cuts which will remove material that bring the model
        to the correct shape
        """

        num_angles = len(self.view_angles)

        # for each possible angle, generate a set of admissable cuts
        for angle_idx in range(num_angles):
            # angle_idx = 1
            angle = self.view_angles[angle_idx]
        
            # get a flattened version of the stl points from the specific angle
            perspective_array = self.rotate_and_flatten(angle)

            # get alpha shape which contains the set of points. Similar to a concave hull
            alpha_shape = alphashape.alphashape(perspective_array.T, 2)
            alpha_shape = alpha_shape.simplify(self.threshold, preserve_topology=True)
            alpha_shape = np.array(alpha_shape.boundary.xy)

            # calculate the convex hull of the array
            hull = ConvexHull(perspective_array.T)
            hull_points = perspective_array[:,hull.vertices]

            # if two sequential points are concave, remove points until we only have one concave
            # point. Prevents non-admissable cuts
            is_concave = self.get_concave_points(alpha_shape, hull_points)
            while self.contains_double_concave(is_concave):
                alpha_shape, is_concave = self.remove_double_concave(alpha_shape, is_concave)
                is_concave = self.get_concave_points(alpha_shape, hull_points)
                
            # One last checks to re-classify concave points purely based on angles, not convex hull
            is_concave = self.do_last_concave_check(alpha_shape, is_concave)

            # plt.figure()
            # plt.plot(*alpha_shape)
            # plt.plot(*perspective_array, "*b")
            # for test_idx in range(len(is_concave)):
            #     if is_concave[test_idx]:
            #         plt.plot(*alpha_shape[:,test_idx],"or")
            #     else:
            #         plt.plot(*alpha_shape[:,test_idx],"ok")
            # plt.plot(*hull_points)

            cut_lines = self.generate_cut_lines(alpha_shape, is_concave)

            if np.array_equal(alpha_shape[:,[0]], alpha_shape[:,[-1]]):
                alpha_shape = np.delete(alpha_shape, -1, 1)
                is_concave.pop(-1)

            # while plane_idx < len(is_concave) - 1:
            #     if is_concave[plane_idx]:
            #         plt.plot(*cut_lines[:,:,plane_idx], "-r")
            #     elif is_concave[plane_idx+1]:
            #         plt.plot(*cut_lines[:,:,plane_idx], "-r")
            #     else:
            #         plt.plot(*cut_lines[:,:,plane_idx], "-g")
            #         # print("convex")
            #     plane_idx += 1

            # start converting the 2D lines into 3D planes
            planes = []

            # take each 2D line and make it into a 3D plane and then rotate it into global frame
            for line_idx in range(len(cut_lines[0,0,:])):
                cut_line = cut_lines[:,:,line_idx]
                cut_points = np.zeros((3,4))
                cut_points[0,:] = np.concatenate((cut_line[0,:], np.flip(cut_line[0,:])), axis=0)
                cut_points[1,:] = np.array([4, 4, -4, -4])
                cut_points[2,: ] = np.concatenate((cut_line[1,:], np.flip(cut_line[1,:])), axis=0)
                cut_points_adjusted = self.unrotate(angle, cut_points)

                # make the plane into a Plane object by set of bounding coordinates
                planes.append(Plane(cut_points_adjusted))

            # add each plane as a cut. If the point is concave, add the set of two planes as a cut
            
            # plt.figure()

            # print(len(is_concave), len(planes))
            plane_idx = 0
            end_plane = len(planes) 
            while plane_idx < end_plane:
                if is_concave[plane_idx]:
                    # grab plane before and at
                    # plt.plot(*cut_lines[:,:,plane_idx], "-r")
                    # plt.plot(*cut_lines[:,:,plane_idx-1], "-r")
                    cur_plane = planes[plane_idx]
                    prev_plane = planes[plane_idx-1]

                    self.cuts.append(Cut([cur_plane, prev_plane]))

                    if plane_idx == 0:
                        end_plane -= 1
                    plane_idx +=1
                elif is_concave[plane_idx - 1]:
                    # grab plane before and at
                    # plt.plot(*cut_lines[:,:,plane_idx-1], "-r")
                    # plt.plot(*cut_lines[:,:,plane_idx-2], "-r")
                    prev_plane = planes[plane_idx-1]
                    prev_prev_plane = planes[plane_idx-2]

                    self.cuts.append(Cut([prev_plane, prev_prev_plane]))

                    if plane_idx == 0:
                        end_plane -= 1
                    # plane_idx +=1

                else:
                    # plt.plot(*cut_lines[:,:,plane_idx-1], "-g")
                    prev_plane = planes[plane_idx-1]
                    self.cuts.append(Cut([prev_plane]))

                plane_idx += 1
                    
    def do_last_concave_check(self, points: np.ndarray, is_concave: List[bool]) -> List[bool]:
        n_points = len(is_concave) - 1
        for idx in range(n_points):
            if idx == 0:
                prev_point = points[:,[-2]]
            else:
                prev_point = points[:,[idx-1]]
            cur_point = points[:,[idx]]
            next_point = points[:,[idx+1]]
             
            angle = self.angle(cur_point, prev_point, next_point)

            if angle < math.pi:
                is_concave[idx] = 0
            
        is_concave[-1] = is_concave[0]

        return is_concave

    def contains_double_concave(self, is_concave: List[bool]) -> bool:
        """
        Using the is_concave bool list, check if any sequential points are concave
        """
        for idx in range(len(is_concave)-1):
            cur_value = is_concave[idx]
            nex_value = is_concave[idx + 1]
            if cur_value and nex_value:
                return True
        return False

    def get_concave_points(self, vertices: np.ndarray, hull: np.ndarray) -> List[bool]:
        """
        Given a set of points that form the bounds of a polygon, return a list of bool which
        identify which points are the inner point of a concave feature.
        """

        # initialize empty bool list
        is_concave = []

        # get size of alpha_shape
        n_points = len(vertices[0,:]) - 1
        
        # check to see if the point is in the convex hull
        for idx in range(n_points):
            cur_point = vertices[:,[idx]]
            is_concave.append(not np.any(np.isin(hull, cur_point)))
        
        is_concave.append(is_concave[False])

        return is_concave

    def remove_double_concave(self, points: np.ndarray, is_concave: List[bool]
                              ) -> Tuple[np.ndarray, List[bool]]:
        """
        Given a set of points and a list of which ones are concave, remove points until there are
        no consecutive concave points
        """
        # remove last point
        points = points[:,0:-1]
        is_concave = is_concave[:-1]

        # get to good starting points
        for idx, val in enumerate(is_concave):
            if val == 0:
                break
        
        # using new starting points, shift things around
        points = np.concatenate((points[:,idx:], points[:,:idx]), axis=1)
        is_concave = is_concave[idx:] + is_concave[:idx]

        # copy first value to the last spot
        points = np.concatenate((points, points[:,[0]]), axis = 1)
        is_concave.append(is_concave[0])

        # find double concave section
        idx = 0 
        while idx < len(is_concave)-1:
            cur_point = points[:,idx]

            # start of concave, see if it continues
            if is_concave[idx] == 0 and is_concave[idx+1] == 1:
                end_idx = idx+1
                while True:
                    end_idx += 1
                    if is_concave[end_idx] == 0:
                        break
                concave_len = end_idx - idx

                # if the length of concave points are longer than 2
                if concave_len > 2:
                    end_point = points[:,[end_idx]]
                    max_area = 0
                    best_idx = 0
                    # check all the middle points and try to form triangle
                    for mid_idx in range(concave_len):
                        mid_point = points[:,[idx+mid_idx+1]]
                        area = .5*abs(cur_point[0]*(mid_point[1] - end_point[1]) + \
                                      mid_point[0]*(end_point[1] - cur_point[1]) + \
                                      end_point[0]*(cur_point[1] - mid_point[1]))
                        
                        # if possible triangle formed by end points and concave point is the
                        # largest triangle formed, keep it
                        if area > max_area:
                            max_area = area
                            best_idx = idx+mid_idx+1
                    
                    # remove all concave points that are not the one that forms the largest triangle
                    points = np.concatenate((points[:,:idx+1], points[:,[best_idx]], points[:,end_idx:]), axis=1)
                    is_concave = is_concave[0:idx+1] + [is_concave[best_idx]] + is_concave[end_idx:]

            idx +=1

        return points, is_concave

    def generate_cut_lines(self, points: np.ndarray, is_concave: bool) -> np.ndarray:
        """
        Given the set of points which define the alpha shape of a polygon, generate a set of 2D
        lines which represent cuts that follow the edge. 
        """
        cut_lines = np.zeros((2,2,len(points[0,:])-1))
        for idx in range(len(is_concave) - 1):
            cur_point = points[:,[idx]]
            next_point = points[:,[idx+1]]

            if is_concave[idx+1]:
                vec = np.concatenate((cur_point, next_point), axis=1)                
                vec = (vec - vec[:,[1]]) / np.linalg.norm(vec - vec[:,[1]]) * 2 + vec[:,[1]]
                
            elif is_concave[idx]:
                vec = np.concatenate((cur_point, next_point), axis=1)
                vec =  (vec - vec[:,[0]]) / np.linalg.norm(vec - vec[:,[0]]) * 2 + vec[:,[0]]

            else:
                vec = np.concatenate((cur_point, next_point), axis=1)
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

    def generate_view_angles(self, nb_poses: int) -> List[Tuple[float, float]]:
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

        array = np.array([[math.cos(theta), -math.sin(theta), 0],
                          [math.sin(theta), math.cos(theta), 0],
                          [0, 0, 1]]) @ array
        
        array = np.array([[1, 0, 0],
                          [0, math.cos(phi), -math.sin(phi)],
                          [0, math.sin(phi), math.cos(phi)]]) @ array
        
        return array[0:3:2,:]
    
    def unrotate(self, perspective: Tuple[float, float], cut_line: np.ndarray) -> np.ndarray:
        """
        Given a cut plane, unrotate it using the current perspective which contains phi and theta
        """
        phi = perspective[0]
        theta = -perspective[1]

        array =  np.array([[1, 0, 0],
                           [0, math.cos(phi), -math.sin(phi)],
                           [0, math.sin(phi), math.cos(phi)]]) @ cut_line
        
        
        array = np.array([[math.cos(theta), -math.sin(theta), 0],
                          [math.sin(theta), math.cos(theta), 0],
                          [0, 0, 1]]) @ array
        return array
    