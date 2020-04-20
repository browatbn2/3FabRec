from math import atan
import numpy as np

from .transformations import translation_matrix, rotation_matrix, concatenate_matrices


class Camera(object):
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = np.deg2rad(orientation)
        self.near = 1.0
        self.far = 25.0
        self.fov_x = np.deg2rad(130)
        self.fov_y = np.deg2rad(130)
        self.atan_fov_x = atan(self.fov_x / 2.0)
        self.atan_fov_y = atan(self.fov_y / 2.0)

    def lookAt(self, target):
        # tbd
        pass

    def get_perspective_projection(self):
        z_range = (self.far + self.near) / (self.far - self.near)
        perspective_proj = [[-self.atan_fov_x, 0, 0, 0],
                            [0, self.atan_fov_y, 0, 0],
                            [0, 0, -z_range, -2 * (self.far * self.near) / (self.far - self.near)],
                            [0, 0, 1, 0.0]]
        perspective_proj = np.asarray(perspective_proj)
        # t = transformations.concatenate_matrices(objToWorld, worldToCamera)
        return perspective_proj

    def get_orthrographic_projection(self):
        range = self.far - self.near
        a = 2.0 / range
        b = (self.far + self.near) / range
        ortho_proj = [[-1, 0,  0,  0],
                      [0,  1,  0,  0],
                      [0,  0, -a, -b],
                      [0,  0,  1,  0]]
        return np.asarray(ortho_proj)


    def get_view_matrix(self):
        t = translation_matrix(self.position)
        r_x = rotation_matrix(self.orientation[0], [1,0,0])
        r_y = rotation_matrix(self.orientation[1], [0,1,0])
        r_z = rotation_matrix(self.orientation[2], [0,0,1])
        return concatenate_matrices(r_z, r_y, r_x, t)