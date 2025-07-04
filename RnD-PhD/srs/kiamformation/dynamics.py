"""Функции для моделирования динамики КА
ПЕРЕДЕЛАТЬ v_->vrs"""
from config import Variables
from cosmetic import my_print

# >>>>>>>>>>>> Поступательное движение, интегрирование <<<<<<<<<<<<
def get_atm_params(v: Variables, h: float, atm_model: str = None) -> tuple:
    """NASA модель: https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html (якорная точка: 25 км)
    ПНБО модель: https://www.energia.ru/ktt/archive/2022/04-2022/101-111.pdf (120-600 км)
    COESA62, COESA76 модели: библиотека poliastro
    :param v: Объект класса Variables
    :param h: Высота
    :param atm_model: Модель атмосферы (необязательный параметр)
    :return: (ρ, T, P): плотность, температура, давление (Внимание! Для ПНБО (ρ, None, None)"""
    import numpy as np
    atm_model = v.ATMOSPHERE_MODEL if atm_model is None else atm_model
    rho, T, p = None, None, None
    if atm_model == 'NASA':
        if h > 25e3:
            T = -131.21 + 0.00299 * h
            p = 2.488 * ((T + 273.1) / 216.6) ** -11.388
        elif h > 11e3:
            T = -56.46
            p = 22.65 * np.exp(1.73 - 0.000157 * h)
        else:
            T = 15.04 - 0.00649 * h
            p = 101.29 * (T + 273.1) / 288.08
        rho = p / (0.2869 * (T + 273.1))
    if atm_model == 'ПНБО':
        A = 2.519e-10
        H_m = 200e3
        H_0 = 290e3
        K_0 = 0.26
        a_1 = 100e3
        a_2 = 141.13e3
        n_0 = 6.34
        n_01 = 4.25
        n_02 = 4.37
        if h > 290e3:
            n = n_0 + K_0 * ((h - H_0) / a_1) ** n_01 - ((h - H_0) / a_2) ** n_02
        else:
            n = n_0 + K_0 * ((H_0 - h) / a_1) ** n_01
        rho = A * (H_m / h) ** n
    if atm_model in ['COESA62', 'COESA76']:
        from astropy import units as u
        from poliastro.earth.atmosphere import COESA62, COESA76
        coesa = COESA62() if atm_model == 'COESA62' else COESA76()
        T, p, rho = coesa.properties(h * u.m)
        T = T.value
        p = p.value
        rho = rho.value
    return rho, T, p

def get_geopotential_acceleration(vrs: Variables, r, v, w, mu=None):
    """Возвращает ускорение КА от притяжения Земли. Внимание! При ('hkw' in _vrs.SOLVER) ускорение в ОСК, иначе ИСК!"""
    from flexmath import setvectype, norm
    if 'hkw' in vrs.SOLVER:
        return setvectype([-2*w*v[2],
                           -w**2*r[1],
                           3*w**2*r[2] + 2*w*v[0]])
    return mu * r / norm(r) ** 3

def get_aero_drag_acceleration(vrs: Variables, obj, i: int, r, v, rho=None):
    """Возвращает ускорение КА от сопротивления атмосферы.
    Внимание! При параметре vrs.SOLVER='hkw' возвращает ускорение в ОСК, иначе в ИСК!"""
    from my_math import quart2dcm, matrix2angle
    from flexmath import setvectype, norm
    S = quart2dcm(obj.q[i])
    cos_alpha = matrix2angle(S) if obj.name == "FemtoSat" else 1

    if 'hkw' in vrs.SOLVER:
        v_real = v + setvectype([vrs.V_ORB, 0, 0])
        rho = get_atm_params(v=vrs, h=r[2] + vrs.HEIGHT)[0] if rho is None else rho
        vrs.RHO = rho
        return - v_real * norm(v_real) * obj.get_blown_surface(cos_alpha) * rho / obj.mass

def get_full_acceleration(vrs: Variables, obj, i: int, r, v, w=None, mu=None, rho=None):
    """Возвращает вектор силы в ОСК, принимает параметры в ОСК"""
    w = vrs.W_ORB if w is None else w
    mu = vrs.MU if mu is None else mu
    if 'hkw' in vrs.SOLVER:
        force = get_geopotential_acceleration(vrs=vrs, r=r, v=v, w=w, mu=mu)
        if vrs.DYNAMIC_MODEL['aero drag']:
            force += get_aero_drag_acceleration(vrs=vrs, r=r, v=v, obj=obj, i=i, rho=rho)
        return force

def translate_rhs(vrs: Variables, obj, i: int, rv, w=None, mu=None, rho=None):
    """При численном моделировании rv передаётся 1 numpy.ndarray, иначе rv типа tuple"""
    from flexmath import append
    r, v = rv if isinstance(rv, tuple) else (rv[[0, 1, 2]], rv[[3, 4, 5]])
    dr = v
    dv = get_full_acceleration(vrs=vrs, obj=obj, i=i, r=r, v=v, w=w, mu=mu, rho=rho)
    return (dr, dv) if isinstance(rv, tuple) else append(dr, dv)

def rk4_translate(v_: Variables, obj, i: int, dt: float = None, r=None, v=None) -> tuple:
    """Функция работает только с численными переменными"""
    import numpy as np
    dt = v_.dT if dt is None else dt
    r = obj.r_orf[i] if r is None else r
    v = obj.v_orf[i] if v is None else v

    rv = np.append(r, v)
    k1 = translate_rhs(vrs=v_, obj=obj, i=i, rv=rv)
    k2 = translate_rhs(vrs=v_, obj=obj, i=i, rv=rv + k1 * dt / 2)
    k3 = translate_rhs(vrs=v_, obj=obj, i=i, rv=rv + k2 * dt / 2)
    k4 = translate_rhs(vrs=v_, obj=obj, i=i, rv=rv + k3 * dt)
    rv = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return rv[0:3] + r, rv[3:6] + v


# >>>>>>>>>>>> Вращательное движение, интегрирование <<<<<<<<<<<<
def get_torque(v: Variables, obj, q, w, t, i):
    """Вектор внешнего углового ускорения"""
    from flexmath import norm, cross, inv
    q = obj.q if q is None else q
    w = obj.w_brf if w is None else w
    J = obj.J
    U, S, A, R_orb = get_matrices(vrs=v, t=t, obj=obj, n=i, q=q)
    R = A @ R_orb
    # m_grav = np.zeros(3)
    m_grav = 3*v.MU/norm(R)**5 * cross(R, J @ R)
    return inv(J) @ (m_grav - cross(w, J @ w))

def attitude_rhs(vrs: Variables, obj, t: float, i: int, qw):
    """При численном моделировании qw передаётся 1 numpy.ndarray;
    При символьном вычислении qw должен быть типа tuple"""
    from flexmath import quat, float2rational
    from my_math import q_dot
    import numpy as np
    from sympy import Rational
    # q, w = qw if isinstance(qw, tuple) else (np.quaternion(*qw[[0, 1, 2, 3]]), qw[[4, 5, 6]])
    q, w = qw if isinstance(qw, tuple) else (quat(qw[[0, 1, 2, 3]]), qw[[4, 5, 6]])

    # if isinstance(q, np.quaternion):
    #     dq = 1 / 2 * q_dot(q, quat(w))
    # else:
    #     dq = Rational(1, 2) * q_dot(q, quat(w))
    dq = 1 / 2 * q_dot(quat(w), q)
    dq = float2rational(dq, (1, 2))

    dw = get_torque(v=vrs, obj=obj, q=q, w=w, t=t, i=i)
    return (dq, dw) if isinstance(qw, tuple) else np.append(dq.components, dw)

def rk4_attitude(vrs: Variables, obj, t: float, i: int, dt: float = None, q=None, w=None):
    """Функция работает только с численными переменными.
    Если принял на вход q-3, возвращает q-3; аналогично q-4"""
    import numpy as np
    import quaternion
    from my_math import vec2quat

    dt = vrs.dT if dt is None else dt
    q = obj.q[i] if q is None else q
    w = obj.w_brf[i] if w is None else w

    q4 = q if isinstance(q, np.quaternion) else vec2quat(q)

    qw = np.append(quaternion.as_float_array(q4), w)
    k1 = attitude_rhs(vrs=vrs, obj=obj, t=t, i=i, qw=qw)
    k2 = attitude_rhs(vrs=vrs, obj=obj, t=t, i=i, qw=qw + k1 * dt / 2)
    k3 = attitude_rhs(vrs=vrs, obj=obj, t=t, i=i, qw=qw + k2 * dt / 2)
    k4 = attitude_rhs(vrs=vrs, obj=obj, t=t, i=i, qw=qw + k3 * dt)
    qw = dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

    w_anw = w + qw[4:7]  # qw[[4, 5, 6]]
    q_anw = (q4 + np.quaternion(*qw[0:4])).normalized()  # qw[[0, 1, 2, 3]]

    if q_anw.w < 0:
        q_anw *= -1

    return (q_anw, w_anw) if isinstance(q, np.quaternion) else (q_anw.vec, w_anw)


# >>>>>>>>>>>> Перевод между системами координат <<<<<<<<<<<<
def get_matrices(vrs: Variables, t, obj=None, n: int = None, first_init: bool = False, q=None):
    """Функция возвращает матрицы поворота.
    Инициализируется в dymancis.py, используется в spacecrafts, dynamics"""
    from flexmath import arctan, sqrt, setvectype, sin, cos, tan, cross, norm, float2rational, get_same_type_conversion
    from my_math import quart2dcm, vec2unit
    from sympy import Matrix
    q = obj.q[n] if q is None else q
    E = t * vrs.W_ORB  # Эксцентрическая аномалия
    if vrs.ECCENTRICITY == 0:
        f = E
    else:
        f = 2 * arctan(sqrt((1 + vrs.ECCENTRICITY) / (1 - vrs.ECCENTRICITY)) * tan(E / 2))  # Истинная аномалия
    A = quart2dcm(q)
    if 'hkw' in vrs.SOLVER or first_init:
        U = setvectype([[0, 1, 0],  # Поворот к экваториальной плоскости
                        [0, 0, 1],
                        [1, 0, 0]]) @ \
            setvectype([[cos(f), sin(f), 0],  # Разница между истинной аномалией и местной
                        [-sin(f), cos(f), 0],
                        [0, 0, 1]]) @ \
            setvectype([[1, 0, 0],  # Поворот к плоскости орбиты
                        [0, cos(vrs.INCLINATION), sin(vrs.INCLINATION)],
                        [0, -sin(vrs.INCLINATION), cos(vrs.INCLINATION)]])
        translation = vrs.P / (1 + vrs.ECCENTRICITY * cos(f))
        U = float2rational(U, (2, 1), (1, 2), (1, 1))
    else:
        e_z = vec2unit(vrs.ANCHOR.r_irf[0])
        e_x = vec2unit(vrs.ANCHOR.v_irf[0])
        e_y = vec2unit(cross(e_z, e_x))
        e_x = vec2unit(cross(e_y, e_z))
        U = setvectype([e_x, e_y, e_z])
        translation = norm(vrs.ANCHOR.r_irf[0])
    if isinstance(A, Matrix):
        U.simplify()
    S = A @ U.T
    if isinstance(A, Matrix):
        S.simplify()
    tmp = get_same_type_conversion(U)
    R_orb = U.T @ tmp([0, 0, translation])
    return U, S, A, R_orb

def i_o(a, vrs: Variables, U, vec_type: str):
    """Инерциальная -> Орбитальная"""
    from flexmath import get_same_type_conversion
    t = get_same_type_conversion(a)
    if len(a.shape) == 1 and vec_type == "r":
        return U @ a - t([0, 0, vrs.ORBIT_RADIUS])
    if len(a.shape) == 1 and vec_type == "v":
        return U @ a - t([vrs.V_ORB, 0, 0])
    if len(a.shape) == 1 and vec_type == "w":
        return U @ (a - vrs.W_ORB_VEC_IRF)
    if len(a.shape) == 2:
        return U @ a @ U.T
    raise ValueError(f"Необходимо подать вектор или матрицу! Тип вектора {vec_type} должен быть из [r, v, w]")

def o_i(a, vrs: Variables, U, vec_type: str):
    """Орбитальная -> Инерциальная"""
    from flexmath import get_same_type_conversion
    t = get_same_type_conversion(a)
    if len(a.shape) == 1 and vec_type == "r":
        return U.T @ (a + t([0, 0, vrs.ORBIT_RADIUS]))
    if len(a.shape) == 1 and vec_type == "v":
        return U.T @ (a + t([vrs.V_ORB, 0, 0]))
    if len(a.shape) == 1 and vec_type == "w":
        return U.T @ a + vrs.W_ORB_VEC_IRF
    if len(a.shape) == 2:
        return U.T @ a @ U
    raise ValueError(f"Необходимо подать вектор или матрицу! Тип вектора {vec_type} должен быть из [r, v, w]")


# >>>>>>>>>>>> Класс динамики кубсатов и чипсатов <<<<<<<<<<<<
class PhysicModel:
    from spacecrafts import FemtoSat, CubeSat, Anchor

    def __init__(self, f: FemtoSat, c: CubeSat, a: Anchor, v: Variables):
        from pandas import DataFrame
        from gnc_systems import KalmanFilter

        # Неизменные параметры
        self.t = 0.
        self.iter = 0
        self.v = v
        self.c = c
        self.f = f
        self.a = a
        self.spacecrafts_cd = [self.c, self.f]
        self.spacecrafts_all = [self.a, self.c, self.f]
        self.time2nav = 0.

        # Инициализация фильтра
        self.k = KalmanFilter(f=f, c=c, p=self)

        # Инициализация траектории kiam-astro
        self.jd0, self.tr = None, None
        if 'kiamastro' in self.v.SOLVER:
            self.kiam_init()

        # Запись параметров
        self.record = DataFrame()

    def kiam_init(self):
        import numpy as np
        from kiam_astro import kiam
        from kiam_astro.trajectory import Trajectory
        self.jd0 = kiam.juliandate(2024, 1, 1, 0, 0, 0)  # (год, месяц, день, чч, мм, сс)
        self.tr = [[Trajectory(initial_state=np.zeros(6), initial_time=0, initial_jd=self.jd0, variables='rv',
                               system='gcrs', units_name='earth') for _ in range(obj.n)]
                   for obj in [self.c, self.f, self.a]]
        for j, obj in enumerate(self.spacecrafts_all):
            for i in range(obj.n):
                s0 = np.append(obj.r_irf[i] / (kiam.units('earth')['DistUnit'] * 1e3),
                               obj.v_irf[i] / (kiam.units('earth')['VelUnit'] * 1e3))
                self.tr[j][i] = Trajectory(initial_state=s0, initial_time=0, initial_jd=self.jd0, variables='rv',
                                           system='gcrs', units_name='earth')
                self.tr[i][j].set_model(variables='rv', model_type='nbp', primary='earth',
                                        sources_list=[] + ['j2'] if self.v.DYNAMIC_MODEL['j2'] else [] +
                                                     ['atm'] if self.v.DYNAMIC_MODEL['aero drag'] else [])
                self.tr[i][j].model['data']['jd_zero'] = self.jd0
                self.tr[i][j].model['data']['mass'] = self.f.mass
                self.tr[i][j].model['data']['area'] = self.f.size[0] * self.f.size[1]
                self.tr[i][j].model['data']['order'] = 0  # order of the Moon's gravity field
                self.tr[i][j].propagate(tof=self.v.TIME/self.v.SEC_IN_RAD, npoints=int(self.v.TIME//self.v.dT))
        my_print(f"kiam-astro Time: {self.tr[0][0].times[-1]} ({self.tr[0][0].times[-1]/2/np.pi} оборотов)\n"
                 f"kiam-astro Points: {self.v.TIME/self.v.dT}", if_print=self.v.IF_ANY_PRINT)
        if self.v.IF_ANY_SHOW:
            self.tr[0][0].show(variables='3d', language='rus')
            self.tr[1][0].show(variables='3d', language='rus')

    # Шаг по времени
    def time_step(self):
        import numpy as np
        from gnc_systems import navigate, guidance
        from primary_info import measure_antennas_power, measure_magnetic_field

        self.iter += 1
        self.t += self.v.dT

        # Движение системы
        for j, obj in enumerate(self.spacecrafts_all):
            for i in range(obj.n):
                U, _, A, _ = get_matrices(vrs=self.v, t=self.t, obj=obj, n=i)

                # Вращательное движение
                if obj != self.a:
                    obj.q[i], obj.w_brf[i] = rk4_attitude(vrs=self.v, obj=obj, i=i, t=self.t)

                # Поступательное движение
                if 'rk4' in self.v.SOLVER:
                    obj.r_orf[i], obj.v_orf[i] = rk4_translate(v_=self.v, obj=obj, i=i)
                    obj.r_irf[i] = o_i(vrs=self.v, a=obj.r_orf[i], U=U, vec_type='r')
                    obj.v_irf[i] = o_i(vrs=self.v, a=obj.v_orf[i], U=U, vec_type='v')
                elif 'kiamastro' in self.v.SOLVER:
                    from kiam_astro import kiam
                    obj.r_irf[i] = np.array([self.tr[j][i].states[ii][self.iter - 1]
                                             for ii in range(3)]) * kiam.units('earth')['DistUnit'] * 1e3
                    obj.v_irf[i] = np.array([self.tr[j][i].states[ii + 3][self.iter - 1]
                                             for ii in range(3)]) * kiam.units('earth')['VelUnit'] * 1e3
                    tr_time = self.tr[j][i].times[self.iter-1] * self.v.SEC_IN_RAD
                    U, _, _, _ = get_matrices(vrs=self.v, t=tr_time, obj=obj, n=i)
                    obj.r_orf[i] = i_o(vrs=self.v, a=obj.r_irf[i], U=U, vec_type='r')
                    obj.v_orf[i] = i_o(vrs=self.v, a=obj.v_irf[i], U=U, vec_type='v')
                else:
                    raise ValueError(f"Solver is to be changed! {self.v.SOLVER} not in {self.v.SOLVERS}")

                U, _, A, _ = get_matrices(vrs=self.v, t=self.t, obj=obj, n=i)
                obj.w_irf[i] = A.T @ obj.w_brf[i]
                obj.w_orf[i] = i_o(vrs=self.v, a=obj.w_irf[i], U=U, vec_type='w')

        # Комплекс первичной информации
        noise = np.sqrt(self.v.KALMAN_COEF['r'])
        measure_antennas_power(c=self.c, f=self.f, vrs=self.v, noise=noise, produce=True, p=self, estimated_params=[])
        measure_magnetic_field(c=self.c, f=self.f, vrs=self.v, noise=noise)

        # Изменение режимов работы
        guidance(v=self.v, c=self.c, f=self.f, earth_turn=self.t * self.v.W_ORB / 2 / np.pi)

        # Навигация чипсатов
        if self.v.IF_NAVIGATION:
            navigate(k=self.k, if_correction=self.time2nav <= 0.)
            self.time2nav = self.v.dT - self.v.dT if self.time2nav <= 0 else self.time2nav - self.v.dT

        # Запись параметров
        self.do_report()

    def do_report(self):
        import numpy as np

        i_t = self.iter
        d = self.record
        d.loc[i_t, f'i'] = self.iter
        d.loc[i_t, f't'] = self.t
        n_tmp = len(self.v.MEASURES_VECTOR)
        d.loc[i_t, f'MEASURES_VECTOR N'] = n_tmp
        d.loc[i_t, [f'MEASURES_VECTOR {i}' for i in range(n_tmp)]] = self.v.MEASURES_VECTOR
        for obj in self.spacecrafts_all:
            d.loc[i_t, f'{obj.name} n'] = obj.n
            for i_n in range(obj.n):
                for v in ['r', 'q', 'v', 'w']:
                    tmp = {'r': [obj.r_irf[i_n], obj.r_orf[i_n]],
                           'v': [obj.v_irf[i_n], obj.v_orf[i_n]],
                           'q': [obj.q[i_n].vec],
                           'w': [obj.w_irf[i_n], obj.w_orf[i_n]]}[v]
                    for i_fr, frame in enumerate(['irf', 'orf'] if v != 'q' else ['irf']):  # Кватернионы только в ИСК
                        for i_r, c in enumerate('xyz'):
                            d.loc[i_t, f'{obj.name} {v} {c} {frame} {i_n}'] = tmp[i_fr][i_r]

        for obj in [self.f]:
            for i_n in range(obj.n):
                if obj.operating_mode[i_n] != "lost":  # Иначе заполняется Null (в plot в self.v.NO_LINE_FLAG)
                    r_orf_estimation = self.k.get_estimation(i_f=i_n, v='r orf')
                    w_brf_estimation = self.k.get_estimation(i_f=i_n, v='w brf')
                    q_irf_estimation = self.k.get_estimation(i_f=i_n, v='q-3 irf')
                    r_orf = self.f.r_orf[i_n]
                    w_brf = self.f.w_brf[i_n]
                    q_irf = self.f.q[i_n].vec

                    w_orf, w_orf_estimation = [], []  # Чтобы PyCharm не ругался
                    if self.v.NAVIGATION_ANGLES:
                        U, _, A, _ = get_matrices(vrs=self.v, t=self.t, obj=obj, n=i_n)
                        w_irf = A.T @ w_brf
                        w_orf = i_o(a=w_irf, vrs=self.v, vec_type='w', U=U)
                        w_irf_estimation = A.T @ w_brf_estimation
                        w_orf_estimation = i_o(a=w_irf_estimation, vrs=self.v, vec_type='w', U=U)

                    d.loc[i_t, f'{obj.name} KalmanPosEstimation r {i_n}'] = np.linalg.norm(r_orf_estimation)
                    d.loc[i_t, f'{obj.name} KalmanPosError r {i_n}'] = np.linalg.norm(r_orf_estimation - r_orf)
                    if self.v.NAVIGATION_ANGLES:
                        d.loc[i_t, f'{obj.name} KalmanSpinError w {i_n}'] = np.linalg.norm(w_brf_estimation - w_brf)
                        d.loc[i_t, f'{obj.name} KalmanQuatError q {i_n}'] = np.linalg.norm(q_irf_estimation - q_irf)
                    for i_r, c in enumerate('xyz'):
                        d.loc[i_t, f'{obj.name} KalmanPosEstimation {c} {i_n}'] = r_orf_estimation[i_r]
                        d.loc[i_t, f'{obj.name} KalmanPosError {c} {i_n}'] = r_orf_estimation[i_r] - r_orf[i_r]
                        if self.v.NAVIGATION_ANGLES:
                            d.loc[i_t, f'{obj.name} RealSpin BRF {c} {i_n}'] = w_brf[i_r]
                            d.loc[i_t, f'{obj.name} RealSpin ORF {c} {i_n}'] = w_orf[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinEstimation BRF {c} {i_n}'] = w_brf_estimation[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinEstimation ORF {c} {i_n}'] = w_orf_estimation[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinError BRF {c} {i_n}'] = w_brf_estimation[i_r] - w_brf[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinError ORF {c} {i_n}'] = w_orf_estimation[i_r] - w_orf[i_r]
                            d.loc[i_t, f'{obj.name} RealQuat {c} {i_n}'] = q_irf[i_r]
                            d.loc[i_t, f'{obj.name} KalmanQuatEstimation {c} {i_n}'] = q_irf_estimation[i_r]
                            d.loc[i_t, f'{obj.name} KalmanQuatError {c} {i_n}'] = q_irf_estimation[i_r] - q_irf[i_r]
