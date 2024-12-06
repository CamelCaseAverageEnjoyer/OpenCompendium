from datetime import datetime, timedelta
import numpy as np

# Писать следующим образом:
# ---> в _tiny_functions.py

# ---------------> Координатные приколы <--------------- #
def coord2const(X: np.ndarray, w: float) -> list:
    """Функция констант С для ХКУ движения
    :param X: Фазовый 6-вектор
    :param w: Угловая скорость орбиты
    """
    return [2 * X[2] + X[3] / w,
            X[5] / w,
            -3 * X[2] - 2 * X[3] / w,
            X[0] - 2 * X[5] / w,
            X[4] / w,
            X[1]]

def trajectory(w: float, C: list, t:float) -> np.ndarray:
    """
    :param w: Угловая скорость орбиты
    :param C: Константы ХКУ
    :param t: Время, секунды
    """
    return np.array([-3*C[0]*w*t + 2*C[1]*np.cos(w*t) - 2*C[2]*np.sin(w*t) + C[3],
                     C[4]*np.sin(w*t) + C[5]*np.cos(w*t),
                     2*C[0] + C[1]*np.sin(w*t) + C[2]*np.cos(w*t),
                     -3*C[0]*w - 2*C[1]*w*np.sin(w*t) - 2*C[2]*w*np.cos(w*t),
                     C[4]*w*np.cos(w*t) - C[5]*w*np.sin(w*t),
                     C[1]*w*np.cos(w*t) - C[2]*w*np.sin(w*t)])

def Hill_equation(y: np.ndarray, W: float):
    """
    :return: 6-вектор dy    
    """
    return np.array([y[3], y[4], y[5], -2*W*y[5], -W**2*y[1], 2*W*y[3] + 3*W**2*y[2]])

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
"""Сравнить с этим
def euler2rot_matrix(a: float, b: float, g: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]) @ \
        np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])"""

def q_dot(L1, L2):
    """Функция является кватернионным умножением \n
    Кватернион L1,L2 передаются векторами длины 4 \n
    Возвращает кватернион L[0]..L[3]."""
    return np.array([L1[0] * L2[0] - L1[1] * L2[1] - L1[2] * L2[2] - L1[3] * L2[3],
                     L1[0] * L2[1] + L1[1] * L2[0] + L1[2] * L2[3] - L1[3] * L2[2],
                     L1[0] * L2[2] + L1[2] * L2[0] + L1[3] * L2[1] - L1[1] * L2[3],
                     L1[0] * L2[3] + L1[3] * L2[0] + L1[1] * L2[2] - L1[2] * L2[1]])

def quart2dcm(L):
    """Функция ищет матрицу поворота из кватерниона поворота \n
    Кватернион L передаётся вектором длины 4 \n
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

def orbitalMotionKeplerian(mu, p, epsilon, phi, omega, inc, t_pi, t_cur, approx):
    """
    :param mu: Гравитационный параметр
    :param p: Высота орбиты
    :param epsilon: Экцентриситет
    :param phi: Долгота восходящего узла
    :param omega: Аргумент перицентра
    :param inc: Наклонение
    :param t_pi: Перицентровое время
    :param t_cur: Текущее время
    :param approx: Первый шаг для метода Ньютона
    :return: (r, v, D) - положение и скорость спутника
    """    
    if epsilon < 1:
        a = p/(1 - epsilon**2)
        b = np.sqrt(a*p)
        M = np.sqrt(mu/a**3)*(t_cur - t_pi)
        misclosure = 1
        D = approx
        while misclosure > 1e-11:
            D = (epsilon*np.sin(D) - epsilon*np.cos(D)*D + M)/(1 - epsilon*np.cos(D))
            # D = D - (D - epsilon*np.sin(D) - M)/(1 - epsilon*np.cos(D))
            misclosure = abs(D - epsilon*np.sin(D) - M)
        # print(D - epsilon*np.sin(D) - M)  # пометка Ульяны
        r1 = np.array([a*(np.cos(D) - epsilon), b*np.sin(D), 0])
        cosTheta = (np.cos(D) - epsilon)/(1 - epsilon*np.cos(D))
        sinTheta = np.sqrt(1 - epsilon**2)*np.sin(D)/(1 - epsilon*np.cos(D))
        Vr = np.sqrt(mu/p)*epsilon*sinTheta
        Vn = np.sqrt(mu/p)*(1 + epsilon*cosTheta)
        v1 = np.array([Vr*cosTheta - Vn*sinTheta, Vr*sinTheta + Vn*cosTheta, 0])
    else:
        raise ValueError(f"Экцентриситет {epsilon}! Неэллиптичная траектория! Мы к этому не готовы!")
        
    A1 = np.array([[np.cos(phi), np.sin(phi), 0],
                   [-np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])
        
    A2 = np.array([[1, 0, 0],
                   [0, np.cos(inc), np.sin(inc)],
                   [0, -np.sin(inc), np.cos(inc)]])
      
    A3 = np.array([[np.cos(omega), np.sin(omega), 0],
                   [-np.sin(omega), np.cos(omega), 0],
                   [0, 0, 1]])
           
    B = A1.T @ A2.T @ A3.T
    
    r = B @ r1
    v = B @ v1
    return r, v, D

# ---------------> Магнитная фигня <--------------- #
def mag_field(mu_0, r, Mu):
    """
    :param mu_0:
    :param r: Радиус-вектор в какой-то СК
    :param Mu:
    :return: B - напряжённость магнитного поля
    """
    R = np.linalg.norm(r)
    B = (mu_0/(4*np.pi*R**5))*(3*Mu*r**2 - R**2*Mu)
    return B

def Inclined_field(t: float, r: np.ndarray, B0, t0scal: datetime, k0, DEG_TO_RAD: float):
    """
    :param t:
    :param r:
    :param B0: был global КАКАЯ РАЗМЕРНОСТЬ
    :param t0scal: был global
    :param k0: был global
    :param DEG_TO_RAD: был global
    :retrun: Bmag
    """
    # Получаем текущее время
    tcur = t0scal + timedelta(milliseconds=round(1000 * t))
    # Вычисляем юлианскую дату в днях
    JD = juliandate(tcur.date())  # Сверено с комплексом
    
    # Перевод вектора направления диполя от гринвической системы к инерциальной текущей даты
    theta = (280.46061837 + 360.98564736629 * (JD - 2451545.0)) * DEG_TO_RAD
    
    k = [k0[0] * np.cos(theta) - k0[1] * np.sin(theta),
         k0[0] * np.sin(theta) + k0[1] * np.cos(theta),
         k0[2]]
    return B0 * (k -3 * (k * r) * r)  # АААААААААААААААААААААААААААА?

def Bmag_dip_orb(u: float, incl: float, B0, w02: float) -> tuple:
    """ Расчёты ведутся в ОСК
    :param incl: был global
    :param B0: был global
    :param w02: был global
    """
    Bdip = np.array([np.sin(incl) * np.cos(u), 
                     np.cos(incl), 
                     -2 * np.sin(incl) * np.sin(u)])
    dBdip = np.array([-w02 * np.sin(incl) * np.sin(u), 
                      0, 
                      -2 * w02 * np.sin(incl) * np.cos(u)])

    # B=B0*.5*(1+(1+3*sin(incl)^2)^.5)*Bdip.T
    B = B0 * Bdip  # ЧТО ЗДЕСЬ ПРОИСХОДИТ, В0 ЭТО ЧИСЛО ИЛИ ВЕКТОР?
    dB = B0 * dBdip  # А ВЫХОДНОЙ РЕЗАЛТ ЭТО ЧИСЛА ИЛИ ВЕКТОРА?
    return B, dB

def Bmag_orb(t: float, u: float, incl: float, F, B0, t0scal, k0, DEG_TO_RAD: float) -> tuple:
    """ Орбита - круговая
    :param incl: был global
    :param F: был global
    :return: Bmag, dB
    """

    # Считаем радиус-вектор (единичный) аппарата
    r_sat = np.array([np.cos(u), 
                      np.cos(incl) * np.sin(u), 
                      np.sin(incl) * np.sin(u)])

    # Матрица перехода от системы, связанной с орбитой, к орбитальной
    G = np.array([[np.cos(u), -np.sin(u), 0]
                  [np.sin(u), np.cos(u), 0]
                  [0, 0, 1]])

    # Задаем магнитное поле - наклонный диполь
    Bmag = Inclined_field(t=t, r=r_sat, B0=B0, t0scal=t0scal, k0=k0, DEG_TO_RAD=DEG_TO_RAD)

    # Перевод ИСК Земли - ИСК орбиты - орбитальная - связанная
    Bmag = G * F * Bmag
    dB = Bmag.copy()
    Bmag = np.array([Bmag[1], Bmag[2], Bmag[0]])  # господи что здесь происходит
    return Bmag, dB

def dipole_force(mu_0, r: np.ndarray, mu, Mu):
    """Это магнитный или электрический? Че делать с Mu*mu"""
    R = np.linalg.norm(r)
    F = (3 * mu_0 / (4 * np.pi * R**5)) * (Mu*mu*r + Mu*r*mu + mu*r*Mu - (5/R**2)*(Mu*r)*(mu*r)*r)
    return F

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
           R = A*[X(j,i), Y(j,i), Z(j,i)]
           X[j, i] = R[0]
           Y[j, i] = R[1]
           Z[j, i] = R[2]
    return X, Y, Z

def function magnetic_dipole(F, d, mu1) -> np.ndarray:
    """
    :param F:
    :param d:
    :param mu1:
    :return: MU
    """
    import sympy as sym
    mu_0 = 4*sym.pi*1e-7  # Проницаемость свободного пространства
    mu2_x, mu2_y, mu2_z = sym.var("mu2_x mu2_y mu2_z")
    eqn1 = sym.Eq((3*mu_0/(4*pi*(norm(d))^5))*((mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(1)+(mu1'*d)*mu2_x+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(1)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(1)), F)
    eqn2 = sym.Eq((mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(2)+(mu1'*d)*mu2_y+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(2)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(2), 0)
    eqn3 = sym.Eq((mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(3)+(mu1'*d)*mu2_z+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(3)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(3), 0)
    sol = nsolve([eqn1, eqn2, eqn3], [mu2_x mu2_y mu2_z])
    return np.array([sol.mu2_x, sol.mu2_y, sol.mu2_z])


# ---------------> Методы Рунге-Кутты <--------------- #
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
    # acceleration_J2 = delta*r/norm(r)^5*(5*z^2/norm(r)^2 - 1) - 2*delta/norm(r)^5*[0, 0, z]
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
