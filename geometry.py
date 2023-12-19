"""
File to hold classes Plane and Cut
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Tuple
import trimesh
import copy
import math


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

        norm_vec = -(np.cross(vec_2, vec_1)) / np.linalg.norm(np.cross(vec_2, vec_1))

        return norm_vec

    def plot(self, color: str = "b") -> None:
        """
        Given a 3D plot, plot the planes
        """
        def drawpoly(xco, yco, zco):
            verts = [list(zip(xco, yco, zco))]
            temp = Poly3DCollection(verts)
            temp.set_facecolor(color)
            temp.set_alpha(.2)
            temp.set_edgecolor('k')

            plt.gca().add_collection3d(temp)
        
        drawpoly(*self.corners)

class Cut:
    """
    Class to contain methods needed to define an admissable cut that removes material
    """
    def __init__(self, slices: List[Plane]) -> None:
        self.slices = slices
        self.is_convex = len(self.slices) == 1
        self.mesh = self.generate_mesh()
        self.is_cut = False

    def plot(self) -> None:
        """
        Plot plane or planes that define the cut
        """
        for slice in self.slices:
            if self.is_convex:
                slice.plot(color="g")
            else:
                slice.plot(color="r")
        
    def generate_mesh(self) -> None:
        """
        Each cut can be represented as a mesh. If we remove the overlap region of the mesh and the
        model, we preform a cut. 
        """
        if self.is_convex:
            slice = self.slices[0]
            vert_1 = slice.corners

            # create a new set of vertices that are parallel offset from the original plane 
            vert_2 = vert_1 - np.reshape(slice.norm_vector,(3,1)) * 2*np.ones((1,4))

            # create a rectangular mesh using the original vertices and the offset vertices
            new_vertices = np.concatenate((vert_1.T, vert_2.T))
            new_faces = np.array([[0,1,2], [0,2,3], [4,1,0], [1,4,5],
                                  [6,5,4], [7,6,4], [6,7,2], [3,2,7],
                                  [5,2,1], [2,5,6], [0,3,4], [3,7,4]])
            cut_mesh = trimesh.Trimesh(new_vertices, new_faces)
        else:
            # get the vertices from the two slices that form a wedge shape
            slice_1 = self.slices[0]
            slice_2 = self.slices[1]
            corners_1 = slice_1.corners
            corners_2 = slice_2.corners

            # create a new triangle prism mesh that connects the two planes
            new_vertices = np.concatenate((corners_1.T,corners_2.T))
            new_faces = np.array([[4,1,0], [1,2,3], [0,1,3], [2,1,4],
                                  [2,4,7], [6,7,4], [6,4,0], [7,3,2]])
            cut_mesh = trimesh.Trimesh(new_vertices, new_faces)

        return cut_mesh

    def calculate_cut_surface_area(self, mesh: trimesh.Trimesh) -> float:
        """
        Calculate the surface area of the cut given a model and the cut
        """
        # create a copy model to mess around with
        mesh = copy.deepcopy(mesh)

        # if cut is convex
        if self.is_convex:
            plane = self.slices[0]

            # get points of plane intersection
            intersection = mesh.section(plane.norm_vector, plane.point)
            if intersection is None:
                return 0
            intersection = intersection.to_planar()[0].vertices.T
            intersection = self.order_points(intersection)

            return self.calculate_area(intersection)

        else:
            plane_1 = self.slices[0]
            plane_2 = self.slices[1]

            # get points of plane_1 intersection
            intersection = mesh.section(plane_1.norm_vector, plane_1.point)

            if intersection is None:
                area_1 = 0

            else:
                vertices = intersection.vertices

                corner_1 = plane_1.corners[:,[0]]
                corner_2 = plane_1.corners[:,[3]]

                ray_vec = corner_2 - corner_1

                ray_intersect = mesh.ray.intersects_location(corner_1.T, ray_vec.T)[0]

                vertices = np.concatenate((vertices, ray_intersect), axis = 0).T

                # reduce vertices by other plane
                a_barrier, b_barrier, c_barrier, d_barrier = self.calculate_plane(plane_2.corners)

                valid_points = []
                for idx in range(len(vertices[0,:])):
                    cur_point = vertices[:,idx]
                    x_val = cur_point[0]
                    y_val = cur_point[1]
                    z_val = cur_point[2]

                    in_cut = a_barrier * x_val + b_barrier * y_val + c_barrier * z_val + d_barrier - .0001 < 0

                    valid_points.append(in_cut)

                vertices = vertices.T[np.array(valid_points)].T

                points = self.unrotate_plane(vertices, plane_1.norm_vector)

                points = self.order_points(points)

                area_1 = self.calculate_area(points)

            # get points of plane_2 intersection
            intersection = mesh.section(plane_2.norm_vector, plane_2.point)

            if intersection is None:
                area_2 = 0
            else:

                vertices = intersection.vertices

                vertices = np.concatenate((vertices, ray_intersect), axis = 0).T

                # reduce vertices by other plane
                a_barrier, b_barrier, c_barrier, d_barrier = self.calculate_plane(plane_1.corners)

                valid_points = []
                for idx in range(len(vertices[0,:])):
                    cur_point = vertices[:,idx]
                    x_val = cur_point[0]
                    y_val = cur_point[1]
                    z_val = cur_point[2]

                    in_cut = a_barrier * x_val + b_barrier * y_val + c_barrier * z_val + d_barrier - .0001 < 0

                    valid_points.append(in_cut)

                vertices = vertices.T[np.array(valid_points)].T

                points = self.unrotate_plane(vertices, plane_2.norm_vector)

                points = self.order_points(points)

                area_2 = self.calculate_area(points)

            return area_1 + area_2
        
    def calculate_cut_surface_area_2(self, mesh: trimesh.Trimesh) -> float:
        """
        Calculate the surface area of the cut given a model and the cut
        """
        # create a copy model to mess around with
        mesh = copy.deepcopy(mesh)

        # if cut is convex
        if self.is_convex:
            plane = self.slices[0]

            # get points of plane intersection
            intersection = mesh.section(plane.norm_vector, plane.point)
            if intersection is None:
                return 0

            return intersection.to_planar(normal = plane.norm_vector)[0].area
        else:
            plane_1 = self.slices[0]
            plane_2 = self.slices[1]

            # get area of first cut
            mesh_1 = mesh.slice_plane(plane_2.point, -plane_2.norm_vector, cap=True)
            intersection = mesh_1.section(plane_1.norm_vector, plane_1.point)

            if intersection is None:
                area_1 = 0
            else:
                area_1 = intersection.to_planar()[0].area

            # get area of second cut
            mesh_2 = mesh.slice_plane(plane_1.point, -plane_1.norm_vector, cap=True)
            intersection = mesh_2.section(plane_2.norm_vector, plane_2.point)

            if intersection is None:
                area_2 = 0
            else:
                area_2 = intersection.to_planar()[0].area
            return area_1 + area_2




            

    def calculate_area(self, points: np.ndarray) -> float:
        """
        Given a set of 2d points, calculate the area that they bound
        """
        x_val = points[0,:]
        y_val = points[1,:]

        return 0.5*np.abs(np.dot(x_val,np.roll(y_val,1))-np.dot(y_val,np.roll(x_val,1)))

    def order_points(self, points: np.ndarray) -> np.ndarray:
        """
        Given that points may not be in order, reorder based on angle
        """
        x_vals = points[0,:]
        y_vals = points[1,:]

        x_center = np.average(x_vals)
        y_center = np.average(y_vals)

        angles = np.arctan2(x_vals-x_center, y_vals-y_center)
        indices = np.argsort(angles)

        points = points.T[indices].T

        return points
    
    def calculate_plane(self, points: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Given 3 points in 3d space, return the formula of the form Ax + By +Cz + d = 0
        """
        x_1 = points[0,0]
        x_2 = points[0,1]
        x_3 = points[0,2]
        y_1 = points[1,0]
        y_2 = points[1,1]
        y_3 = points[1,2]
        z_1 = points[2,0]
        z_2 = points[2,1]
        z_3 = points[2,2]

        a1 = x_2 - x_1
        b1 = y_2 - y_1
        c1 = z_2 - z_1
        a2 = x_3 - x_1
        b2 = y_3 - y_1
        c2 = z_3 - z_1
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * x_1 - b * y_1 - c * z_1)

        return(a,b,c,d)
    
    def unrotate_plane(self, points: np.ndarray, norm_vec: np.ndarray) -> np.ndarray:
        """
        If points are on a plane, un-rotate them so they all lie on the xy axis
        """
        M = norm_vec
        N = np.array([0,0,1])

        cos_theta = np.dot(M,N)
        axis = np.cross(M,N)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        s = math.sqrt(1-cos_theta**2)
        C = 1-cos_theta

        # print(x,y,z)

        R = np.array([[x*x*C + cos_theta, x*y*C-z*s, x*z*C+y*s],
                        [y*x*C+z*s, y*y*C+cos_theta, y*z*C-x*s],
                        [z*x*C-y*s, z*y*C+x*s, z*z*C+cos_theta]])
        
        new_array =  R @ points
        return new_array[0:2,:]
    
    def calculate_removed_volume(self, mesh: trimesh.Trimesh) -> float:
        """
        Calculate the area of material that will be removed
        """

        # create a copy model to mess around with
        mesh = copy.deepcopy(mesh)

        cut_mesh = self.mesh

        intersection = trimesh.boolean.boolean_manifold([mesh, cut_mesh], operation="intersection")

        return intersection.volume



