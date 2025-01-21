import numpy as np

def q2dcm(q):
    w, x, y, z = q
    return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w],
                     [2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w],
                     [2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2]])
