import numpy as np
from scipy import spatial
from stl import mesh
import vedo

def get_cube(dims: np.ndarray, A: np.ndarray, tns: np.ndarray):
    h = 0.5
    r = [[-h, -h, -h, h, -h, -h, h, h, -h, h, h, h, h, h, -h, h, h, h, -h, -h, -h, h, -h, -h],
         [-h, -h, -h, -h, -h, h, -h, -h, h, h, -h, h, h, h, h, h, -h, h, h, h, -h, -h, -h, h],
         [-h, h, -h, -h, -h, -h, -h, h, -h, -h, -h, -h, -h, h, h, h, h, h, -h, h, h, h, h, h]]
    r = np.array(r).T

    for i in range(len(r)):
        r[i] = r[i] * dims  # stretching
        r[i] = A @ r[i]  # rotation
        r[i] = r[i] + tns  # translation

    vertices = []
    for i in range(len(r)):
        vertices.append(r[i])
    vertices = np.array(vertices)

    hull = spatial.ConvexHull(vertices)
    faces = hull.simplices
    myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            myramid_mesh.vectors[i][j] = vertices[f[j], :]
    # return myramid_mesh

    verts_temp, faces_temp = [], []
    for i in range(len(myramid_mesh.v0)):
        verts_temp.append(myramid_mesh.v0[i])
        verts_temp.append(myramid_mesh.v1[i])
        verts_temp.append(myramid_mesh.v2[i])
        faces_temp.append([i * 3, i * 3 + 1, i * 3 + 2])
    return vedo.Mesh([verts_temp, faces_temp]).clean()
    

