import numpy as np


def ortogonal_vec(r1, r2):
    """Функция возвращает какой-то ортогональный вектор к отрезку r1-r2"""
    n = r1 - r2
    if np.linalg.norm(my_cross(n, [1., 0., 0.])) > 1e-4:
        b = my_cross(n, [1., 0., 0.])
        b /= np.linalg.norm(b)
    else:
        b = [0., 1., 0.]
    t = my_cross(n, b)
    return t / np.linalg.norm(t)

def local_tensor_of_rod(r1, r2):
    """Тензор инерции стрежня в СтСК, без учёта массы"""
    J_xx = (r1[1] ** 2 + r1[1] * r2[1] + r2[1] ** 2 + r1[2] ** 2 + r1[2] * r2[2] + r2[2] ** 2) / 3
    J_yy = (r1[0] ** 2 + r1[0] * r2[0] + r2[0] ** 2 + r1[2] ** 2 + r1[2] * r2[2] + r2[2] ** 2) / 3
    J_zz = (r1[0] ** 2 + r1[0] * r2[0] + r2[0] ** 2 + r1[1] ** 2 + r1[1] * r2[1] + r2[1] ** 2) / 3
    J_xy = - (2 * r1[0] * r1[1] + r1[0] * r2[1] + r2[0] * r1[1] + 2 * r2[0] * r2[1]) / 6
    J_yz = - (2 * r1[1] * r1[2] + r1[1] * r2[2] + r2[1] * r1[2] + 2 * r2[1] * r2[2]) / 6
    J_zx = - (2 * r1[0] * r1[2] + r1[0] * r2[2] + r2[0] * r1[2] + 2 * r2[0] * r2[2]) / 6
    return np.array([[J_xx, J_xy, J_zx],
                     [J_xy, J_yy, J_yz],
                     [J_zx, J_yz, J_zz]])

def local_tensor_of_point(m, r):
    anw = np.zeros((3, 3))
    for v in range(3):
        for w in range(3):
            anw[v][w] = m * (int(v == w) * np.linalg.norm(r) ** 2 - r[v] * r[w])
    return anw

def velocity_spread(u, k_u):
    """Добавляет к вектору u шум с относительной величиной k_u"""
    tmp = np.random.rand(len(u))
    return np.array(u) + tmp / np.linalg.norm(tmp) * np.linalg.norm(u) * k_u

def get_v0(o, id_app, t):
    """Функция начального приближения по ХКУ"""
    w = o.w_hkw
    x0, y0, z0 = o.a.r[id_app]
    x1, y1, z1 = o.b_o(o.a.target[id_app])
    den = -3 * t * w * np.sin(t * w) - 8 * np.cos(t * w) + 8
    num1 = w * (6 * t * w * z0 * np.sin(t * w) - x0 * np.sin(t * w) + x1 * np.sin(t * w) + 14 * z0 * np.cos(t * w) -
                14 * z0 - 2 * z1 * np.cos(t * w) + 2 * z1)
    num2 = w * (3 * t * w * z0 * np.cos(t * w) - 3 * t * w * z1 - 2 * x0 * np.cos(t * w) + 2 * x0 +
                2 * x1 * np.cos(t * w) - 2 * x1 - 4 * z0 * np.sin(t * w) + 4 * z1 * np.sin(t * w))
    return np.array([num1 / den,
                     w * (-y0 * np.cos(t * w) + y1)/np.sin(t * w),
                     num2 / den])

def get_c_hkw(r, v, w):
    """Возвращает константы C[0]..C[5] движения Хилла-Клохесси-Уилтштира"""
    return [2*r[2] + v[0]/w, v[2]/w, -3*r[2] - 2*v[0]/w, r[0] - 2*v[2]/w, v[1]/w, r[1]]

def r_hkw(C, w, t):
    """Возвращает вектор координат в момент времени t; \n
    Уравнения движения Хилла-Клохесси-Уилтштира; \n
    Константы C передаются массивом C[0]..C[5]; \n
    Частота w, время t должны быть скалярными величинами."""
    return np.array([-3 * C[0] * w * t + 2 * C[1] * np.cos(w * t) - 2 * C[2] * np.sin(w * t) + C[3],
                     C[5] * np.cos(w * t) + C[4] * np.sin(w * t),
                     2 * C[0] + C[2] * np.cos(w * t) + C[1] * np.sin(w * t)])

def v_hkw(C, w, t):
    """Возвращает вектор скоростей в момент времени t; \n
    Уравнения движения Хилла-Клохесси-Уилтштира; \n
    Константы C передаются массивом C[0]..C[5]; \n
    Частота w, время t должны быть скалярными величинами."""
    return np.array([-3 * C[0] * w - 2 * w * C[1] * np.sin(w * t) - 2 * w * C[2] * np.cos(w * t),
                     w * C[4] * np.cos(w * t) - w * C[5] * np.sin(w * t),
                     -w * C[2] * np.sin(w * t) + w * C[1] * np.cos(w * t)])

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

def q_dot(L1, L2):
    """Функция является кватернионным умножением; \n
    Кватернион L1,L2 передаются векторами длины 4; \n
    Возвращает кватернион L[0]..L[3]."""
    return np.array([L1[0] * L2[0] - L1[1] * L2[1] - L1[2] * L2[2] - L1[3] * L2[3],
                     L1[0] * L2[1] + L1[1] * L2[0] + L1[2] * L2[3] - L1[3] * L2[2],
                     L1[0] * L2[2] + L1[2] * L2[0] + L1[3] * L2[1] - L1[1] * L2[3],
                     L1[0] * L2[3] + L1[3] * L2[0] + L1[1] * L2[2] - L1[2] * L2[1]])

def clip(a, bot, top):
    if a < bot:
        return bot
    if a > top:
        return top
    return a

def my_atan2(c, s):
    """Возвращает угол в радианах из косинуса и синуса"""
    if c > 0 and s > 0:
        return np.arccos(c)
    if c > 0 >= s:
        return np.arcsin(s)
    if c <= 0 < s:
        return np.arccos(c)
    return np.pi - np.arcsin(s)

def my_cross(a, b):
    """Функция векторного произведения"""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])

def matrix_from_vector(v):
    """Функция возвращает матрицу как оператор векторного умножения слева"""
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])

def flatten(lst):
    """Функция берёт 2D массив, делает 1D"""
    return [item for sublist in lst for item in sublist]

def polar2dec(v, phi, theta):
    """Функция переводит полярные координаты в декартовы"""
    return np.array([v * np.cos(phi) * np.cos(theta),
                     v * np.sin(phi) * np.cos(theta),
                     v * np.sin(theta)])

def kronecker(a, b, tolerance=1e-6):
    """Функция является функцией кронокера"""
    tmp = abs(np.linalg.norm(a - np.array(b)))
    return 1 if tmp < tolerance else 0

def print_time(t0, simple=False):
    from mylibs.im_sample import okonchanye
    if simple:
        t = t0
    else:
        t = t0.seconds
    s = t % 60
    t = int(t/60)
    m = t % 60
    h = int(t/60)
    if h > 0:
        return f"{h} час{okonchanye(h)}, {m} минут, {s} сенунд"
    else:
        if m > 0:
            return f"{m} минут, {s} сенунд"
        else:
            return f"{s} сенунд"
