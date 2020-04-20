from utils.M3D.camera import Camera
from utils.M3D.mesh import Mesh
from utils.M3D import transformations
import numpy as np
import cv2



def draw_head_pose(img, pose, color=(0,0,0.4)):

    def calc_screen_coords(normalized_coords):
        screen_coords = normalized_coords.copy()
        screen_coords[:,0] *= display_size[0]
        screen_coords[:,1] *= display_size[1]
        screen_coords[:,0] += display_size[0]/2
        screen_coords[:,1] += display_size[1]/2
        return screen_coords

    display_size = img.shape[:2]

    cam = Camera((0, 0, -5), (0, 0, 0))
    projection_matrix = cam.get_perspective_projection()
    view_matrix = cam.get_view_matrix()

    mesh = Mesh()
    mesh.position = (0, 0, 0)
    mesh.orientation = (pose[0], -pose[1], -pose[2])
    mesh.vertices = [
        (-1,  1,  0),
        ( 1,  1,  0),
        ( 1, -1,  0),
        (-1, -1,  0),

        # (0,  0, 1),
        # (0,  0, 1),
        # (0,  0, 1),
        # (0,  0, 1),
        (-1,  1, 1),
        ( 1,  1, 1),
        ( 1, -1, 1),
        (-1, -1, 1)
    ]
    lines = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),

        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),

        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]
    obj = np.asarray(mesh.vertices, dtype=np.float32)
    # obj[0:4, :2] *= 0.4
    # obj[4:, :2] *= 0.5
    # obj[4:, 2] *= 2
    obj[0:4, :2] *= 0.0
    obj[4:, :2] *= 0.25
    obj[4:, 2] *= 3
    obj = np.hstack((obj, np.ones((obj.shape[0], 1))))

    model_matrix = mesh.get_model_matrix()
    transform_matrix = transformations.concatenate_matrices(model_matrix)
    view_projection_matrix = transformations.concatenate_matrices(projection_matrix, view_matrix)

    mesh.world_coords = transform_matrix.dot(obj.T).T

    mesh.clip_coords = view_projection_matrix.dot(mesh.world_coords.T).T
    mesh.clip_coords[:, 0] = mesh.clip_coords[:, 0] / mesh.clip_coords[:, 3]
    mesh.clip_coords[:, 1] = mesh.clip_coords[:, 1] / mesh.clip_coords[:, 3]
    mesh.clip_coords[:, 2] = mesh.clip_coords[:, 2] / mesh.clip_coords[:, 3]
    mesh.screen_coords = calc_screen_coords(mesh.clip_coords)

    x = mesh.screen_coords[:, 0].astype(int)
    y = mesh.screen_coords[:, 1].astype(int)

    # bottom
    for st, nd in lines[:4]:
        cv2.line(img, (x[st], y[st]), (x[nd], y[nd]), np.array(color), thickness=1, lineType=cv2.LINE_AA)
    # lines
    for st, nd in lines[8:]:
        # (0.2,0.2,0.75)
        cv2.line(img, (x[st], y[st]), (x[nd], y[nd]), np.array(color)*1.5, thickness=1, lineType=cv2.LINE_AA)
    # top
    for st, nd in lines[4:8]:
        cv2.line(img, (x[st], y[st]), (x[nd], y[nd]), np.array(color)*2.0, thickness=1, lineType=cv2.LINE_AA)

