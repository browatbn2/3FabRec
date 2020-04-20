import numpy as np
from .transformations import translation_matrix, rotation_matrix, concatenate_matrices

class Mesh(object):

    def __init__(self):
        self.vertices = []
        self.world_coords = []
        self.screen_coords = []
        self.vertex_uv_coords = []
        self.texture = None

        self.vertex_normals = []
        self.world_normals = []

        self.faces = []
        self.face_normals = []

        self.position = [0,0,0]
        self.orientation = [0,0,0]

        self.color = (255,255,255)
        # r = 255 * (0.25 + (idx % len(faces)) * 0.75 / len(faces))
        # g = 255 * (0.5 + (idx % len(faces)) * 0.5 / len(faces))
        # b = 255 * 1-(0.0 + (idx % len(faces)) * 1.0 / len(faces))

    def get_model_matrix(self):
        trans = translation_matrix(self.position)
        r_x = rotation_matrix(self.orientation[0], [1,0,0])
        r_y = rotation_matrix(self.orientation[1], [0,1,0])
        r_z = rotation_matrix(self.orientation[2], [0,0,1])
        return concatenate_matrices(trans, r_z, r_y, r_x)


def load_model_from_json(jsonFile):
    import json
    meshes = []
    with open(jsonFile) as fp:
        jsonObject = json.load(fp)

        materials = {mat['name']:mat for mat in jsonObject['materials']}

        for jsonMesh in jsonObject['meshes']:
            m = Mesh()
            vertices = jsonMesh['vertices']
            indices = jsonMesh['indices']
            verticesStep = 6 + jsonMesh['uvCount']*2

            for i in range(0, len(vertices), verticesStep):
                m.vertices.append(vertices[i:i+3])
                m.vertex_normals.append(vertices[i+3:i+3+3])
                m.vertex_uv_coords.append(vertices[i+3+3:i+3+3+2])

            for i in range(0, len(indices), 3):
                m.faces.append(indices[i:i+3])

            m.position = jsonMesh['position']
            m.vertex_normals = np.array(m.vertex_normals)
            m.vertex_uv_coords = np.array(m.vertex_uv_coords)
            if 'materialId' in jsonMesh:
                mId = jsonMesh['materialId']
                jsonMaterial = materials[mId]
                try:
                    filename = jsonMaterial['diffuseTexture']['name']
                    m.texture = Texture(os.path.join('models', filename))
                except KeyError:
                    pass
                try:
                    m.color = np.array(jsonMaterial['ambient'])
                    m.color = np.array((1,1,1))

                except:
                    pass

            meshes.append(m)

    return meshes
