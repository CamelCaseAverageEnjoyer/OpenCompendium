# ---> в _tiny_functions.py

def get_rotation_matrix(theta: float, ax: str) -> np.ndarray:
    if ax in "0Xx":
        return np.matrix([[1, 0, 0],
                          [0, m.cos(theta),-m.sin(theta)],
                          [0, m.sin(theta), m.cos(theta)]])
    if ax in "1Yy":
        return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])
    if ax in "2Zz":
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta) , 0],
                          [0, 0, 1]])
    return None

def get_rotation_matrix_by_multiple_rotations(angles: list, sequence: str) -> np.ndarray:
    """Надо проверить что в прямом, но не обратном порядке, и что матрицы не надо транспонировать"""
    if len(angles) != len(sequence):
        print(f"В функции get_rotation_matrix_by_multiple_rotations разные длины аргументов: {len(angles)} и {len(sequence)}")
    M = np.eye(3)
    for i in range(len(angles)):
        M = get_rotation_matrix(theta=angles[i], ax=sequence[i]) @ M
    return M
        

def EarthRotation(X, Y, Z, t: float) -> tuple:
    """
    :return: (X, Y, Z)
    """
    # Z[0, :] = -Z[1, :]
    psi = -360 * np.pi / 180 / 24 / 60 / 60 * t
    theta = 0
    phi = 0
    A = get_rotation_matrix_by_multiple_rotations(angles=[psi, theta, phi], sequence="ZXZ")
    for j in range(len(X[0, :])):
        for i in range(len(X[0, :])):
           R = A*[X(j,i);Y(j,i);Z(j,i)]
           X[j, i] = R[0]
           Y[j, i] = R[1]
           Z[j, i] = R[2]
    return X, Y, Z

