"""Функции для моделирования динамики КА
ПЕРЕДЕЛАТЬ v_->vrs"""
from datetime import datetime

from primary_info import *
from gnc_systems import *
from my_plot import *
from symbolic import get_same_type_conversion, numerical_and_symbolic_polymorph
import quaternion

# >>>>>>>>>>>> Задание движения Хилла-Клохесси-Уилтшира <<<<<<<<<<<<
@numerical_and_symbolic_polymorph(trigger_var=(0, 'r'), trigger_type=np.ndarray, trigger_out=list)
def get_c_hkw(r, v, w, **kwargs):
    """Возвращает константы C[0]..C[5] движения Хилла-Клохесси-Уилтштира"""
    return [2*r[2] + v[0]/w,
            v[2]/w,
            -3*r[2] - 2*v[0]/w,
            r[0] - 2*v[2]/w,
            v[1]/w,
            r[1]]

@numerical_and_symbolic_polymorph(trigger_var=(0, 'C'), trigger_type=list, trigger_out=np.array)
def r_hkw(C, w, t, **kwargs):
    """Возвращает радиус-вектор в ОСК в момент времени t"""
    cos, sin = kwargs['cos'], kwargs['sin']
    return [-3 * C[0] * w * t + 2 * C[1] * cos(w * t) - 2 * C[2] * sin(w * t) + C[3],
            C[5] * cos(w * t) + C[4] * sin(w * t),
            2 * C[0] + C[2] * cos(w * t) + C[1] * sin(w * t)]

@numerical_and_symbolic_polymorph(trigger_var=(0, 'C'), trigger_type=list, trigger_out=np.array)
def v_hkw(C, w, t, **kwargs):
    """Возвращает вектор скоростей в ОСК в момент времени t"""
    cos, sin = kwargs['cos'], kwargs['sin']
    return [-3 * C[0] * w - 2 * w * C[1] * sin(w * t) - 2 * w * C[2] * cos(w * t),
            w * C[4] * cos(w * t) - w * C[5] * sin(w * t),
            w * C[2] * sin(w * t) + w * C[1] * cos(w * t)]

def get_rand_c(v: Variables) -> list:
    """(quaternion or quaternion_but_i_dont_give_a_fuck)"""  # Чего? Что это значит?
    r_spread, v_spread, _ = v.RVW_ChipSat_SPREAD
    return get_c_hkw(r=np.random.uniform(-r_spread, r_spread, 3),
                     v=np.random.uniform(-v_spread, v_spread, 3), w=v.W_ORB)

# >>>>>>>>>>>> Поступательное движение, интегрирование <<<<<<<<<<<<
def get_atm_params(v: Variables, h: float, atm_model: str = None) -> tuple:
    """NASA модель: https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html (якорная точка: 25 км)
    ПНБО модель: https://www.energia.ru/ktt/archive/2022/04-2022/101-111.pdf (120-600 км)
    COESA62, COESA76 модели: библиотека poliastro
    :param v: Объект класса Variables
    :param h: Высота
    :param atm_model: Модель атмосферы (необязательный параметр)
    :return: (ρ, T, P): плотность, температура, давление (Внимание! Для ПНБО (ρ, None, None)"""
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

@numerical_and_symbolic_polymorph(trigger_var=(1, 'r'), trigger_type=np.ndarray, trigger_out=np.array)
def get_geopotential_acceleration(vrs: Variables, r, v, w, mu, **kwargs):
    """Возвращает ускорение КА от притяжения Земли. Внимание! При ('hkw' in _vrs.SOLVER) ускорение в ОСК, иначе ИСК!"""
    norm = kwargs['norm']
    if 'hkw' in vrs.SOLVER:
        return [-2 * w * v[2],
                -w**2 * r[1],
                2 * w * v[0] + 3 * w**2 * r[2]]
    return mu * r / norm(r) ** 3

@numerical_and_symbolic_polymorph(trigger_var=(3, 'r'), trigger_type=np.ndarray, trigger_out=np.array)
def get_aero_drag_acceleration(vrs: Variables, obj: Apparatus, i: int, r, v, rho=None, **kwargs):
    """Возвращает ускорение КА от сопротивления атмосферы.
    Внимание! При параметре vrs.SOLVER='hkw' возвращает ускорение в ОСК, иначе в ИСК!"""
    norm = kwargs['norm']

    S = quart2dcm(obj.q[i])
    cos_alpha = matrix2angle(S) if obj.name == "FemtoSat" else 1
    # rho = get_atm_params(v=vrs, h=obj.r_orf[i][2] + vrs.HEIGHT)[0]

    if 'hkw' in vrs.SOLVER:
        v_real = v + kwargs['vec_type']([vrs.V_ORB, 0, 0])
        rho = get_atm_params(v=vrs, h=r[2] + vrs.HEIGHT)[0] if rho is None else rho
        return - v_real * norm(v_real) * obj.get_blown_surface(cos_alpha) * rho / 2 / obj.mass

def get_full_acceleration(vrs: Variables, obj: Apparatus, i: int, r, v, w=None, mu=None, rho=None):
    """Возвращает вектор силы в ОСК, принимает параметры в ОСК"""
    w = vrs.W_ORB if w is None else w
    mu = vrs.MU if mu is None else mu
    if 'hkw' in vrs.SOLVER:
        force = get_geopotential_acceleration(vrs=vrs, r=r, v=v, w=w, mu=mu)
        if vrs.DYNAMIC_MODEL['aero drag']:
            force += get_aero_drag_acceleration(vrs=vrs, r=r, v=v, obj=obj, i=i, rho=rho)
        return force

@numerical_and_symbolic_polymorph(trigger_var=(3, 'rv'), trigger_type=np.ndarray, trigger_out=lambda x: x)
def translate_rhs(vrs: Variables, obj: Apparatus, i: int, rv, w=None, mu=None, rho=None, **kwargs):
    """При численном моделировании rv передаётся 1 numpy.ndarray, иначе rv типа tuple"""
    append = kwargs['append']
    r, v = rv if isinstance(rv, tuple) else (rv[[0, 1, 2]], rv[[3, 4, 5]])
    dr = v
    dv = get_full_acceleration(vrs=vrs, obj=obj, i=i, r=r, v=v, w=w, mu=mu, rho=rho)
    return (dr, dv) if isinstance(rv, tuple) else append(dr, dv)

def rk4_translate(v_: Variables, obj: Union[CubeSat, FemtoSat], i: int, dt: float = None, r=None, v=None) -> tuple:
    """Функция работает только с численными переменными"""
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
@numerical_and_symbolic_polymorph(trigger_var=(3, 'w'), trigger_type=np.ndarray, trigger_out=lambda x: x)
def get_torque(v: Variables, obj: Apparatus, q, w,  **kwargs):
    """Вектор внешнего углового ускорения"""
    return np.zeros(3)

@numerical_and_symbolic_polymorph(trigger_var=(4, 'qw'), trigger_type=np.ndarray, trigger_out=lambda x: x)
def attitude_rhs(v: Variables, obj: Apparatus, t: float, i: int, qw, **kwargs):
    """При численном моделировании qw передаётся 1 numpy.ndarray;
    При символьном вычислении qw должен быть типа tuple"""
    inv, append, quat = kwargs['inv'], kwargs['append'], kwargs['quat']

    q, w = qw if isinstance(qw, tuple) else (np.quaternion(*qw[[0, 1, 2, 3]]), qw[[4, 5, 6]])

    # U, S, A, R_orb = get_matrices(v=v, t=t, obj=obj, n=i)

    e = get_torque(v=v, obj=obj, q=q, w=w)
    dq = 1 / 2 * q_dot(q, quat(w))
    # J = A.T @ obj.J @ A
    J = obj.J
    dw = - (inv(J) @ (my_cross(w, J @ w))) + e

    return (dq, dw) if isinstance(qw, tuple) else np.append(dq.components, dw)

def rk4_attitude(v_: Variables, obj: Union[CubeSat, FemtoSat], t: float, i: int, dt: float = None, q=None, w=None):
    """Функция работает только с численными переменными.
    Если принял на вход q-3, возвращает q-3; аналогично q-4"""
    dt = v_.dT if dt is None else dt
    q = obj.q[i] if q is None else q
    w = obj.w_irf[i] if w is None else w

    q4 = q if isinstance(q, np.quaternion) else vec2quat(q)

    qw = np.append(quaternion.as_float_array(q4), w)
    k1 = attitude_rhs(v=v_, obj=obj, t=t, i=i, qw=qw)
    k2 = attitude_rhs(v=v_, obj=obj, t=t, i=i, qw=qw + k1 * dt / 2)
    k3 = attitude_rhs(v=v_, obj=obj, t=t, i=i, qw=qw + k2 * dt / 2)
    k4 = attitude_rhs(v=v_, obj=obj, t=t, i=i, qw=qw + k3 * dt)
    qw = dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

    w_anw = w + qw[[4, 5, 6]]
    q_anw = (q4 + np.quaternion(*qw[[0, 1, 2, 3]])).normalized()

    if q_anw.w < 0:
        q_anw *= -1

    return (q_anw, w_anw) if isinstance(q, np.quaternion) else (q_anw.vec, w_anw)


# >>>>>>>>>>>> Перевод между системами координат <<<<<<<<<<<<
@numerical_and_symbolic_polymorph(trigger_var=(1, 't'), trigger_type=(float, int), trigger_out=lambda x: x)
def get_matrices(v: Variables, t, obj: Apparatus = None, n: int = None, first_init: bool = False, q=None, **kwargs):
    """Функция возвращает матрицы поворота.
    Инициализируется в dymancis.py, используется в spacecrafts, dynamics"""
    q = obj.q[n] if q is None else q
    atan, tan, sin, cos, norm, sqrt = \
        kwargs['atan'], kwargs['tan'], kwargs['sin'], kwargs['cos'], kwargs['norm'], kwargs['sqrt']
    E = t * v.W_ORB  # Эксцентрическая аномалия
    if v.ECCENTRICITY == 0:
        f = E
    else:
        f = 2 * atan(sqrt((1 + v.ECCENTRICITY) / (1 - v.ECCENTRICITY)) * tan(E / 2))  # Истинная аномалия
    A = quart2dcm(q)
    if 'hkw' in v.SOLVER or first_init:
        U = kwargs['vec_type']([[0, 1, 0],  # Поворот к экваториальной плоскости
                                [0, 0, 1],
                                [1, 0, 0]]) @ \
            kwargs['vec_type']([[cos(f), sin(f), 0],  # Разница между истинной аномалией и местной
                               [-sin(f), cos(f), 0],
                               [0, 0, 1]]) @ \
            kwargs['vec_type']([[1, 0, 0],  # Поворот к плоскости орбиты
                                [0, cos(v.INCLINATION), sin(v.INCLINATION)],
                                [0, -sin(v.INCLINATION), cos(v.INCLINATION)]])
        translation = v.P / (1 + v.ECCENTRICITY * cos(f))
    else:
        e_z = vec2unit(v.ANCHOR.r_irf[0])
        e_x = vec2unit(v.ANCHOR.v_irf[0])
        e_y = vec2unit(my_cross(e_z, e_x))
        e_x = vec2unit(my_cross(e_y, e_z))
        U = kwargs['vec_type']([e_x, e_y, e_z])
        translation = norm(v.ANCHOR.r_irf[0])
    S = A @ U.T
    R_orb = U.T @ kwargs['vec_type']([0, 0, translation])
    return U, S, A, R_orb

def i_o(a, v: Variables, U, vec_type: str):
    """Инерциальная -> Орбитальная"""
    t = get_same_type_conversion(a)
    if len(a.shape) == 1 and vec_type == "r":
        return U @ a - t([0, 0, v.ORBIT_RADIUS])
    if len(a.shape) == 1 and vec_type == "v":
        return U @ a - t([v.V_ORB, 0, 0])
    if len(a.shape) == 1 and vec_type == "w":
        return U @ (a - v.W_ORB_VEC_IRF)
    if len(a.shape) == 2:
        return U @ a @ U.T
    raise ValueError(f"Необходимо подать вектор или матрицу! Тип вектора {vec_type} должен быть из [r, v, w]")

def o_i(a, v: Variables, U, vec_type: str):
    """Орбитальная -> Инерциальная"""
    t = get_same_type_conversion(a)
    if len(a.shape) == 1 and vec_type == "r":
        return U.T @ (a + t([0, 0, v.ORBIT_RADIUS]))
    if len(a.shape) == 1 and vec_type == "v":
        return U.T @ (a + t([v.V_ORB, 0, 0]))
    if len(a.shape) == 1 and vec_type == "w":
        return U.T @ a + v.W_ORB_VEC_IRF
    if len(a.shape) == 2:
        return U.T @ a @ U
    raise ValueError(f"Необходимо подать вектор или матрицу! Тип вектора {vec_type} должен быть из [r, v, w]")


# >>>>>>>>>>>> Класс динамики кубсатов и чипсатов <<<<<<<<<<<<
class PhysicModel:
    def __init__(self, f: FemtoSat, c: CubeSat, a: Anchor, v: Variables):
        from pandas import DataFrame

        # Неизменные параметры
        self.t = 0.
        self.iter = 0
        self.v = v
        self.c = c
        self.f = f
        self.a = a
        self.spacecrafts_cd = [self.c, self.f]
        self.spacecrafts_all = [self.a, self.c, self.f]
        self.time_begin = datetime.now()

        # Инициализация фильтра
        self.k = KalmanFilter(f=f, c=c, p=self)

        # Инициализация траектории kiam-astro
        self.jd0, self.tr = None, None
        if 'kiamastro' in self.v.SOLVER:
            self.kiam_init()

        # Запись параметров
        self.record = DataFrame()
        # self.do_report()  # Продублировать в конце time_step()

    def kiam_init(self):
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
        self.iter += 1
        self.t = self.iter * self.v.dT

        # Движение системы
        for j, obj in enumerate(self.spacecrafts_all):
            for i in range(obj.n):
                # Вращательное движение
                if obj != self.a and self.v.GAIN_MODEL_C_N + self.v.GAIN_MODEL_F_N > 0:
                    obj.q[i], obj.w_irf[i] = rk4_attitude(v_=self.v, obj=obj, i=i, t=self.t)

                # Поступательное движение
                if 'rk4' in self.v.SOLVER:
                    if np.any(list(self.v.DYNAMIC_MODEL.values())):  # Если J2 или aero drag
                        obj.r_orf[i], obj.v_orf[i] = rk4_translate(v_=self.v, obj=obj, i=i)
                    else:
                        obj.r_orf[i] = r_hkw(obj.c_hkw[i], self.v.W_ORB, self.t)
                        obj.v_orf[i] = v_hkw(obj.c_hkw[i], self.v.W_ORB, self.t)

                    U, _, _, _ = get_matrices(v=self.v, t=self.t, obj=obj, n=i)
                    obj.r_irf[i] = o_i(v=self.v, a=obj.r_orf[i], U=U, vec_type='r')
                    obj.v_irf[i] = o_i(v=self.v, a=obj.v_orf[i], U=U, vec_type='v')
                elif 'kiamastro' in self.v.SOLVER:
                    from kiam_astro import kiam
                    obj.r_irf[i] = np.array([self.tr[j][i].states[ii][self.iter - 1]
                                             for ii in range(3)]) * kiam.units('earth')['DistUnit'] * 1e3
                    obj.v_irf[i] = np.array([self.tr[j][i].states[ii + 3][self.iter - 1]
                                             for ii in range(3)]) * kiam.units('earth')['VelUnit'] * 1e3
                    tr_time = self.tr[j][i].times[self.iter-1] * self.v.SEC_IN_RAD
                    U, _, _, _ = get_matrices(v=self.v, t=tr_time, obj=obj, n=i)
                    obj.r_orf[i] = i_o(v=self.v, a=obj.r_irf[i], U=U, vec_type='r')
                    obj.v_orf[i] = i_o(v=self.v, a=obj.v_irf[i], U=U, vec_type='v')
                else:
                    raise ValueError(f"Поменяй солвер! SOLVER={self.v.SOLVER}, а должен быть среди {self.v.SOLVERS}!")

                obj.w_orf[i] = i_o(v=self.v, a=obj.w_irf[i], U=U, vec_type='w')

        # Комплекс первичной информации
        measure_antennas_power(c=self.c, f=self.f, v=self.v, noise=np.sqrt(self.v.KALMAN_COEF['r']), produce=True,
                               p=self, estimated_params=[])
        measure_magnetic_field(c=self.c, f=self.f, v=self.v, noise=np.sqrt(self.v.KALMAN_COEF['r']))

        # Изменение режимов работы
        guidance(v=self.v, c=self.c, f=self.f, earth_turn=self.t * self.v.W_ORB / 2 / np.pi)

        # Навигация чипсатов
        if self.v.IF_NAVIGATION:
            navigate(k=self.k)

        # Запись параметров
        self.do_report()

    def do_report(self):
        i_t = self.iter
        d = self.record
        d.loc[i_t, f'i'] = self.iter
        d.loc[i_t, f't'] = self.t
        n_tmp = len(self.v.MEASURES_VECTOR)
        d.loc[i_t, f'MEASURES_VECTOR N'] = n_tmp
        d.loc[i_t, [f'MEASURES_VECTOR {i}' for i in range(n_tmp)]] = self.v.MEASURES_VECTOR
        for obj in self.spacecrafts_cd:
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
                    w_irf_estimation = self.k.get_estimation(i_f=i_n, v='w irf')
                    q_irf_estimation = self.k.get_estimation(i_f=i_n, v='q-3 irf')
                    r_orf = self.f.r_orf[i_n]
                    w_irf = self.f.w_irf[i_n]
                    q_irf = self.f.q[i_n].vec

                    w_orf, w_orf_estimation = [], []  # Чтобы PyCharm не ругался
                    if self.v.NAVIGATION_ANGLES:
                        U, _, _, _ = get_matrices(v=self.v, t=self.t, obj=obj, n=i_n)
                        w_orf = i_o(a=w_irf, v=self.v, vec_type='w', U=U)
                        w_orf_estimation = i_o(a=w_irf_estimation, v=self.v, vec_type='w', U=U)

                    d.loc[i_t, f'{obj.name} KalmanPosEstimation r {i_n}'] = np.linalg.norm(r_orf_estimation)
                    d.loc[i_t, f'{obj.name} KalmanPosError r {i_n}'] = np.linalg.norm(r_orf_estimation - r_orf)
                    if self.v.NAVIGATION_ANGLES:
                        d.loc[i_t, f'{obj.name} KalmanSpinError w {i_n}'] = np.linalg.norm(w_irf_estimation - w_irf)
                        d.loc[i_t, f'{obj.name} KalmanQuatError q {i_n}'] = np.linalg.norm(q_irf_estimation - q_irf)
                    for i_r, c in enumerate('xyz'):
                        d.loc[i_t, f'{obj.name} KalmanPosEstimation {c} {i_n}'] = r_orf_estimation[i_r]
                        d.loc[i_t, f'{obj.name} KalmanPosError {c} {i_n}'] = r_orf_estimation[i_r] - r_orf[i_r]
                        if self.v.NAVIGATION_ANGLES:
                            d.loc[i_t, f'{obj.name} RealSpin IRF {c} {i_n}'] = w_irf[i_r]
                            d.loc[i_t, f'{obj.name} RealSpin ORF {c} {i_n}'] = w_orf[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinEstimation IRF {c} {i_n}'] = w_irf_estimation[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinEstimation ORF {c} {i_n}'] = w_orf_estimation[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinError IRF {c} {i_n}'] = w_irf_estimation[i_r] - w_irf[i_r]
                            d.loc[i_t, f'{obj.name} KalmanSpinError ORF {c} {i_n}'] = w_orf_estimation[i_r] - w_orf[i_r]
                            d.loc[i_t, f'{obj.name} RealQuat {c} {i_n}'] = q_irf[i_r]
                            d.loc[i_t, f'{obj.name} KalmanQuatEstimation {c} {i_n}'] = q_irf_estimation[i_r]
                            d.loc[i_t, f'{obj.name} KalmanQuatError {c} {i_n}'] = q_irf_estimation[i_r] - q_irf[i_r]

        d = d.astype({'i': 'int32', 'FemtoSat n': 'int32', 'CubeSat n': 'int32'})