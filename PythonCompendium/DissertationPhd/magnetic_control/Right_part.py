# ---> в 00_tiny_functions.py

def quart2dcm(L):
    """Функция ищет матрицу поворота из кватерниона поворота; \n
    Кватернион L передаётся вектором длины 4; \n
    Возвращает матрицу 3х3."""
    w, x, y, z = L
    A = np.eye(3)
    A[0][0] = 1 - 2 * y ** 2 - 2 * z ** 2
    A[0][1] = 2 * x * y + 2 * z * w
    A[0][2] = 2 * x * z - 2 * y * w
    A[1][0] = 2 * x * y - 2 * z * w
    A[1][1] = 1 - 2 * x ** 2 - 2 * z ** 2
    A[1][2] = 2 * y * z + 2 * x * w
    A[2][0] = 2 * x * z + 2 * y * w
    A[2][1] = 2 * y * z - 2 * x * w
    A[2][2] = 1 - 2 * x ** 2 - 2 * y ** 2
    return A

def Right_part(t: float, X: np.ndarray, Mu, B: np.ndarray, J: np.ndarray, F) -> np.ndarray:
    """ Сейчас нет учёта J2 (и очень харашо)
    :param t:
    :param X: коллокация векторов r, v, q, w
    :param Mu: вроде 3-вектор
    :param B: вроде 3-вектор
    :param J: Тензор инерции
    :param F:
    :return: dX
    """
    r = X[0:3]
    v = X[3:6]
    q = X[6:10]
    w = X[10:13]

    mu=3.986e14  # Перевести в глобальную переменную потом
    R = 6378000  # Перевести в глобальную переменную потом А ТЕПЕРЬ В МЕТРАХ ЕБАТЬ
    delta = 3 / 2 * 1082.8e-6 * mu * R**2
    # acceleration_J2 = delta*r/norm(r)^5*(5*z^2/norm(r)^2 - 1) - 2*delta/norm(r)^5*[0; 0; z];
    acceleration_J2 = np.zeros(3)

    dr = v.copy()
    dv = -mu * r / np.linalg.norm(r)**3 + acceleration_J2 + F
    dq = 0.5 * q_dot(q, [0.] + w.tolist())

    A = quart2dcm(q / np.linalg.norm(q))  # Переход в собственную СК
    M_gr = 3 * mu / np.linalg.norm(r)**5 * my_cross(A @ r, J @ A @ r) 
    # M_gr = np.zeros(3)
    M_mag = my_cross(Mu, A @ B)
    # M_mag = np.zeros(3)

    dw = np.linalg.inv(J) @ (-my_cross(w, J @ w) + M_gr + M_mag)
    return np.append(dr, dv, dq, dw)

