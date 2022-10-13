import os
import numpy as np
from tqdm.auto import tqdm


AU_TO_M = 1.496e11  # Distance to Sun, in meters


def normalize(vector):
    return vector / np.linalg.norm(vector)


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


class Object:
    def __init__(
        self, ambient_color, diffuse_color, specular_color, shininess, reflection, center_xyz
    ):
        self.ambient_color = ambient_color  # Three channel color
        self.diffuse_color = diffuse_color  # Three channel color
        self.specular_color = specular_color  # Three channel color
        self.shininess = shininess  # [0, 1] coefficient
        self.reflection = reflection  # [0, 1] coefficient

        # Not sure using center_xyz across all geometries is a good idea
        # since it means different things for different geometries
        # Also not sure this is the right place for SCREEN_HEIGHT offsets
        self.center_xyz = center_xyz
        # self.center_xyz = [center_xyz[0], center_xyz[1], center_xyz[2] - SCREEN_ELEVATION]
        # print(f"Offsetting along vertical dimension by {SCREEN_ELEVATION} meters...")

    def get_intersection_point(self, ray_origin, ray_direction):
        return None

    def get_normal_vector(self, ray_intersection):
        # Find the normal vector at a given point on the object
        return None


class Sphere(Object):
    def __init__(
        self,
        ambient_color,
        diffuse_color,
        specular_color,
        shininess,
        reflection,
        center_xyz,
        radius,
    ):
        super().__init__(
            ambient_color, diffuse_color, specular_color, shininess, reflection, center_xyz
        )
        self.radius = radius

    def get_intersection_distance(self, ray_origin, ray_direction):
        # This is x2 + y2 = r2, not quadratic formula
        # https://youtu.be/HFPlKQGChpE?t=339
        b = 2 * np.dot(ray_direction, ray_origin - self.center_xyz)  # aka t
        c = np.linalg.norm(ray_origin - self.center_xyz) ** 2 - self.radius**2

        # This is the discriminant. If >0, there exist solutions
        delta = b**2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None

    def get_normal_vector(self, ray_intersection):
        # Find the normal vector at a given point on the object
        vector = ray_intersection - self.center_xyz
        return vector / np.linalg.norm(vector)


class Plane(Object):
    def __init__(
        self,
        ambient_color,
        diffuse_color,
        specular_color,
        shininess,
        reflection,
        center_xyz,
        normal_vec_to_plane=(1, 1, 1),
    ):
        """
        From https://stackoverflow.com/questions/8812073/ray-and-square-rectangle-intersection-in-3d
        """
        super().__init__(
            ambient_color, diffuse_color, specular_color, shininess, reflection, center_xyz
        )
        self.normal_vec_to_plane = normal_vec_to_plane

    def get_intersection_distance(self, ray_origin, ray_direction):
        a, b, c = self.normal_vec_to_plane
        x0, y0, z0 = self.center_xyz
        rx, ry, rz = ray_origin
        vx, vy, vz = ray_direction

        # x, y, z = np.linalg.norm(self.plane_center - ray_origin)

        # Writing this with np.dot is clearer from a math perspective and more complicated as code.
        # They're equivalent though.
        # t = (a * x + b * y + c * z) / (a * vx + b * vy + c * vz)
        # t = (a * (x0 - rx) + b * (y0 - ry) + c * (z0 - rz)) / (a * vx + b * vy + c * vz)
        t = np.dot([a, b, c], [(x0 - rx), (y0 - ry), (z0 - rz)]) / np.dot([a, b, c], [vx, vy, vz])

        # If the denominator is tiny (e.g. the plane's normal and the ray are almost orthogonal), then discard it
        too_parallel = t < 0.0002
        doesnt_count = too_parallel

        return None if doesnt_count else t

    def get_normal_vector(self, ray_intersection):
        # This is probably wrong, but in principle we just want to offset the existing normal vector we've got already
        vector = self.normal_vec_to_plane - ray_intersection
        return vector / np.linalg.norm(vector)


class Disk(Plane):
    def __init__(
        self,
        ambient_color,
        diffuse_color,
        specular_color,
        shininess,
        reflection,
        center_xyz,
        normal_vec_to_plane=(1, 1, 1),
        disk_radius=1,
    ):
        super().__init__(
            ambient_color, diffuse_color, specular_color, shininess, reflection, center_xyz
        )
        self.normal_vec_to_plane = normal_vec_to_plane
        self.disk_radius = disk_radius

    def get_intersection_distance(self, ray_origin, ray_direction):
        a, b, c = self.normal_vec_to_plane
        x0, y0, z0 = self.center_xyz
        rx, ry, rz = ray_origin
        vx, vy, vz = ray_direction

        # x, y, z = np.linalg.norm(self.plane_center - ray_origin)

        t = np.dot([a, b, c], [(x0 - rx), (y0 - ry), (z0 - rz)]) / np.dot([a, b, c], [vx, vy, vz])

        # If the denominator is tiny (e.g. the plane's normal and the ray are almost orthogonal), then discard it
        too_parallel = t < 0.0002
        beyond_radius = (
            np.sqrt(sum([(p - c) ** 2 for p, c in zip(t * ray_direction, self.center_xyz)]))
            > self.disk_radius
        )
        doesnt_count = too_parallel | beyond_radius

        return None if doesnt_count else t


class Rectangle(Plane):
    def __init__(
        self,
        ambient_color,
        diffuse_color,
        specular_color,
        shininess,
        reflection,
        center_xyz,
        normal_vec_to_plane=(1, 1, 1),
        rect_orientation_vec=1,
        rect_dimensions=[2, 4],
    ):
        """
        From https://stackoverflow.com/questions/8812073/ray-and-square-rectangle-intersection-in-3d
        """
        super().__init__(
            ambient_color, diffuse_color, specular_color, shininess, reflection, center_xyz
        )
        self.normal_vec_to_plane = normal_vec_to_plane
        self.disk_radius = disk_radius

    def get_intersection_distance(self, ray_origin, ray_direction):
        a, b, c = self.normal_vec_to_plane
        x0, y0, z0 = self.center_xyz
        rx, ry, rz = ray_origin
        vx, vy, vz = ray_direction

        # x, y, z = np.linalg.norm(self.plane_center - ray_origin)

        t = np.dot([a, b, c], [(x0 - rx), (y0 - ry), (z0 - rz)]) / np.dot([a, b, c], [vx, vy, vz])

        if t < 0.0002:
            return None

        # All the above just gives a plane; now have to restrict it to just do the rectangle
        # Given 𝑝1,𝑝2,𝑝4,𝑝5 vertices of your cuboid, and 𝑝𝑣 the point to test for intersection with the cuboid, compute:
        # 𝑖=𝑝2−𝑝1𝑗=𝑝4−𝑝1𝑘=𝑝5−𝑝1𝑣=𝑝𝑣−𝑝1
        # then, if
        # 0<𝑣⋅𝑖<𝑖⋅𝑖0<𝑣⋅𝑗<𝑗⋅𝑗0<𝑣⋅𝑘<𝑘⋅𝑘
        pv = t * ray_direction  # ?

        return t
