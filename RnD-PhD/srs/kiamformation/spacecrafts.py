"""Функции, связанные с архитектурой КА"""
from my_math import *
from config import Variables
from symbolic import *

# >>>>>>>>>>>> Диаграмма направленности антенн связи <<<<<<<<<<<<
def local_dipole(v: Variables, r, ind: str = 'x', model='quarter-wave monopole'):
    """Возвращает коэффициент усиления от 1-й антенны
    :param v: Объект класса Variables
    :param r: Радиус-вектор между антенной-трансмиттером и антенной-ресивером в ССК
    :param ind: Координата направления антенны-ресивера в ССК
    :param model: """
    if ind not in ['x', 'y', 'z']:
        raise ValueError(f"Координата «{ind}» должна быть среди: [x, y, z]")
    r_12 = r / norm(r)
    r_12 += vec_type([0.03, 0.05, 0], b=r[0]) * v.DISTORTION  # Вручную задаваемое искажение диаграммы направленности
    r_antenna_brf = vec_type([int(ind == 'x'), int(ind == 'y'), int(ind == 'z')], b=r[0])
    
    sin_theta = norm(cross(r_antenna_brf, r_12))
    cos_theta = dot(r_antenna_brf, r_12)
    if model == 'half-wave dipole':
        return cos(pi(r) / 2 * cos_theta) / sin_theta
    if model == 'short dipole':
        return sin_theta**2
    if model == 'quarter-wave monopole':
        return sin_theta**3

def get_gain(v: Variables, obj, r, if_take: bool = False, if_send: bool = False,
             multy_take=None, multy_send=None, gm=None):
    """Возвращает вектор коэффициентов усиления от каждой антенны КА
    :param v: Объект класса Variables
    :param obj: Переменная класса Apparatus
    :param r: Направление сигнала в ССК КА
    :param if_take: Флаг на принятый сигнал
    :param if_send: Флаг на посланный сигнал
    :param multy_take: Разлагается ли сигнал на КА-ресивере (опционально)
    :param multy_send: Разлагается ли сигнал на КА-трасмитере (опционально)
    :param gm: Количество и тип антенн (опционально)"""
    multy_take = v.MULTI_ANTENNA_TAKE if multy_take is None else multy_take
    multy_send = v.MULTI_ANTENNA_SEND if multy_send is None else multy_send
    gm = obj.gain_mode if gm is None else gm

    # Памятка: GAIN_MODES = ['isotropic', '1 antenna', '2 antennas', '3 antennas']
    if gm == v.GAIN_MODES[1]:
        return [local_dipole(v, r, 'x')]
    if gm == v.GAIN_MODES[2]:
        if (if_take and multy_take) or (if_send and multy_send):
            return [local_dipole(v, r, 'x'), local_dipole(v, r, 'y')]
        return [local_dipole(v, r, 'x') + local_dipole(v, r, 'y')]
    if gm == v.GAIN_MODES[3]:
        if (if_take and multy_take) or (if_send and multy_send):
            return [local_dipole(v, r, 'x'), local_dipole(v, r, 'y'), local_dipole(v, r, 'z')]
        return [local_dipole(v, r, 'x') + local_dipole(v, r, 'y') + local_dipole(v, r, 'z')]
    return [1]


# >>>>>>>>>>>> Классы аппаратов <<<<<<<<<<<<
class Apparatus:
    def __init__(self, v: Variables, n: int):
        """Пустой класс КА"""
        from dynamics import get_matrices, o_i, get_c_hkw

        # Общие параметры
        self.name = "No exist"
        self.n = n
        self.mass = 1e10
        self.size = [1., 1., 1.]
        self.c_resist = 0
        self.J = None

        # Индивидуальные параметры движения
        self.w_irf = [np.zeros(3) for _ in range(self.n)]
        self.w_orf = [np.zeros(3) for _ in range(self.n)]
        self.w_brf = [np.zeros(3) for _ in range(self.n)]
        self.q = [np.quaternion(1, 0, 0, 0) for _ in range(self.n)]
        self.r_orf = [np.zeros(3) for _ in range(self.n)]
        self.v_orf = [np.zeros(3) for _ in range(self.n)]
        U, _, _, _ = get_matrices(v=v, t=0, obj=self, n=0, first_init=True)
        self.r_irf = [o_i(v=v, a=self.r_orf[0], U=U, vec_type='r')]
        self.v_irf = [o_i(v=v, a=self.v_orf[0], U=U, vec_type='v')]
        self.c_hkw = [get_c_hkw(self.r_orf[i], self.v_orf[i], v.W_ORB) for i in range(self.n)]

        # Индивидуальные параметры режимов работы
        self.operating_mode = [v.OPERATING_MODES[0] for _ in range(self.n)]

    def get_blown_surface(self, cos_alpha):
        return self.size[0] * self.size[1] * abs(cos_alpha) * self.c_resist

    def update_c(self, v):
        from dynamics import get_c_hkw
        self.c_hkw = [get_c_hkw(self.r_orf[i], self.v_orf[i], v.W_ORB) for i in range(self.n)]

    def update_irf_rv(self, v: Variables, t: float = 0):
        from dynamics import o_i, get_matrices
        for i in range(self.n):
            U, _, _, _ = get_matrices(v=v, t=t, obj=self, n=i)
            self.r_irf[i] = o_i(v=v, a=self.r_orf[i], U=U, vec_type='r')
            self.v_irf[i] = o_i(v=v, a=self.v_orf[i], U=U, vec_type='v')

    def update_irf_w(self, v: Variables, t: float = 0, w_irf: list = None, w_orf: list = None, w_brf: list = None):
        from dynamics import o_i, get_matrices
        w_irf = self.w_irf if w_irf is None else w_irf
        w_orf = self.w_orf if w_orf is None else w_orf
        w_brf = self.w_brf if w_brf is None else w_brf
        for i in range(self.n):
            U, _, A, _ = get_matrices(v=v, t=t, obj=self, n=i)
            w_irf[i] = o_i(a=w_orf[i], v=v, U=U, vec_type='w')
            w_brf[i] = A @ w_irf[i]

    def init_correct_q_v(self, v: Variables, q: list = None):
        q = self.q if q is None else q
        for i in range(self.n):
            if v.SHAMANISM["ClohessyWiltshireC1=0"]:
                self.v_orf[i][0] = - 2 * self.r_orf[i][2] * v.W_ORB
            q[i] = q[i].normalized()
            if q[i].w < 0:
                q[i] *= -1


class Anchor(Apparatus):
    def __init__(self, v: Variables):
        """Класс мнимого КА, центр которого совпадает с центром ОСК"""
        super().__init__(v=v, n=1)
        self.name = "Anchor"

class CubeSat(Apparatus):
    """Класс содержит информацию об n кубсатах модели model_c = 1U/1.5U/2U/3U/6U/12U.
    Все величны представлены в СИ."""
    def __init__(self, v: Variables):
        super().__init__(v=v, n=v.CUBESAT_AMOUNT)

        # Предопределённые параметры
        cubesat_property = {'1U': {'mass': 2.,
                                   'mass_center_error': [0.02, 0.02, 0.02],
                                   'dims': [0.1, 0.1, 0.1135]},
                            '1.5U': {'mass': 3.,
                                     'mass_center_error': [0.02, 0.02, 0.03],
                                     'dims': [0.1, 0.1, 0.1702]},
                            '2U': {'mass': 4.,
                                   'mass_center_error': [0.02, 0.02, 0.045],
                                   'dims': [0.1, 0.1, 0.227]},
                            '3U': {'mass': 6.,
                                   'mass_center_error': [0.02, 0.02, 0.07],
                                   'dims': [0.1, 0.1, 0.3405]},
                            '6U': {'mass': 12.,
                                   'mass_center_error': [4.5, 2., 7.],
                                   'dims': [0.2263, 0.1, 0.366]},
                            '12U': {'mass': 24.,
                                    'mass_center_error': [4.5, 4.5, 7.],
                                    'dims': [0.2263, 0.2263, 0.366]}}

        # Общие параметры
        self.name = "CubeSat"
        self.gain_mode = v.GAIN_MODEL_C
        self.mass = cubesat_property[v.CUBESAT_MODEL]['mass']
        self.size = cubesat_property[v.CUBESAT_MODEL]['dims']
        self.mass_center_error = cubesat_property[v.CUBESAT_MODEL]['mass_center_error']
        self.r_mass_center = np.array([np.random.uniform(-i, i) for i in self.mass_center_error])
        self.c_resist = 1.05
        # Пока что J диагонален
        self.J = np.diag([self.size[1]**2 + self.size[2]**2,
                          self.size[0]**2 + self.size[2]**2,
                          self.size[0]**2 + self.size[1]**2]) * self.mass / 12

        # Индивидуальные параметры движения
        self.r_orf = [v.spread('r', name=self.name) for _ in range(self.n)]
        self.v_orf = [v.spread('v', name=self.name) for _ in range(self.n)]
        self.w_orf = [np.zeros(3) for _ in range(self.n)]

        # Инициализируется автоматически
        self.q = [np.quaternion(1, -1, -1, -1) for _ in range(self.n)]
        self.init_correct_q_v(v=v)
        self.r_irf, self.v_irf, self.w_irf, self.w_irf = [[np.zeros(3) for _ in range(self.n)] for _ in range(4)]
        self.update_irf_rv(v=v, t=0)
        self.update_irf_w(v=v, t=0)
        self.update_c(v=v)

        # Индивидуальные параметры управления
        self.m_self, self.b_env = [[np.zeros(3) for _ in range(self.n)] for _ in range(2)]

        # Прорисовка ножек
        self.legs_x = 0.0085
        self.legs_z = 0.007

class FemtoSat(Apparatus):
    def __init__(self, v: Variables, c: CubeSat):
        """Класс содержит информацию об n фемтосатах.\n
        Все величны представлены в СИ."""
        super().__init__(v=v, n=v.CHIPSAT_AMOUNT)

        # Предопределённые параметры
        chipsat_property = {'KickSat': {'mass': 0.01,
                                        'mass_center_error': [0.001, -0.001],
                                        'dims': [0.03, 0.03]},
                            'Трисат': {'mass': 0.1,
                                       'mass_center_error': [0.005, 0.003],
                                       'dims': [0.4, 0.15]}}

        # Общие параметры
        self.name = "FemtoSat"
        self.gain_mode = v.GAIN_MODEL_F
        self.mass = chipsat_property[v.CHIPSAT_MODEL]['mass']
        self.mass_center_error = chipsat_property[v.CHIPSAT_MODEL]['mass_center_error']
        self.size = chipsat_property[v.CHIPSAT_MODEL]['dims']
        self.c_resist = 1.17
        # Пока что J диагонален
        self.J = np.diag([self.size[1]**2,
                          self.size[0]**2,
                          self.size[0]**2 + self.size[1]**2]) * self.mass / 12
        self.power_signal_full = 0.01
        self.length_signal_full = 0.001

        # Индивидуальные параметры движения
        self.deploy(v=v, c=c, i_c=0)
        self.w_orf_ = [v.spread('w', name=self.name) for _ in range(self.n)]
        # Инициализируется автоматически
        self.r_irf, self.v_irf, self.w_irf, self.w_irf_, self.w_brf, self.w_brf_ = \
            [[np.zeros(3) for _ in range(self.n)] for _ in range(6)]
        self.q, self.q_ = [[np.quaternion(*np.random.uniform(-1, 1, 4)) for _ in range(self.n)] for _ in range(2)]
        self.init_correct_q_v(v=v)
        self.init_correct_q_v(v=v, q=self.q_)
        self.update_irf_rv(v=v, t=0)
        self.update_irf_w(v=v, t=0)
        self.update_irf_w(v=v, t=0, w_irf=self.w_irf_, w_orf=self.w_orf_, w_brf=self.w_brf_)
        self.update_c(v=v)

        # УДАЛИТЬ!!!!
        """from dynamics import r_hkw, v_hkw
        self.c_hkw = [np.array([0, 100, 0, 10, 10, 0])]
        self.r_orf[0] = r_hkw(self.c_hkw[0], v.W_ORB, 0)
        self.v_orf[0] = v_hkw(self.c_hkw[0], v.W_ORB, 0)"""

        # Индивидуальные параметры управления
        self.m_self, self.b_env = [[np.zeros(3) for _ in range(self.n)] for _ in range(2)]

        tol = 1 if v.START_NAVIGATION == v.NAVIGATIONS[0] else v.START_NAVIGATION_TOLERANCE
        tol = 0 if v.START_NAVIGATION == v.NAVIGATIONS[2] else tol

        # Новый формат
        """self.apriori_params = {'r orf': [self.r_orf[i] + [10, 10, 10] for i in range(self.n)],
                               'v orf': [self.v_orf[i] for i in range(self.n)],
                               'w brf': [self.w_brf[i] for i in range(self.n)],
                               'q-3 irf': [self.q[i].vec for i in range(self.n)]}"""
        if v.NAVIGATION_ANGLES:
            self.apriori_params = {'r orf': [self.r_orf[i] * tol + v.spread('r', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'v orf': [self.v_orf[i] * tol + v.spread('v', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'w brf': [self.w_brf[i] * tol + self.w_brf_[i] * (1 - tol) for i in range(self.n)],
                                   'q-3 irf': [self.q[i].vec * tol + self.q_[i].vec * (1 - tol) for i in range(self.n)]}
        else:
            self.apriori_params = {'r orf': [self.r_orf[i] * tol + v.spread('r', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'v orf': [self.v_orf[i] * tol + v.spread('v', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'w brf': [self.w_brf[i] for i in range(self.n)],
                                   'q-3 irf': [self.q[i].vec for i in range(self.n)]}

    def deploy(self, v: Variables, c: CubeSat, i_c: int) -> None:
        """Функция отделения задаёт начальные условия для дочерних КА из материнских КА
        :param v: объект Variables
        :param c: объект CubeSat
        :param i_c: id-номер материнского КА, от которого отделяются дочерние КА
        :return: {'r orf': ..., 'v orf': ..., 'q-3 irf': ..., 'w irf': ...}, где значения - list of np.ndarray
        """
        if v.DEPLOYMENT == v.DEPLOYMENTS[0]:  # No
            self.r_orf = [v.spread('r', name=self.name) for _ in range(self.n)]
            self.v_orf = [v.spread('v', name=self.name) for _ in range(self.n)]
            self.w_orf = [v.spread('w', name=self.name) for _ in range(self.n)]
        elif v.DEPLOYMENT == v.DEPLOYMENTS[1]:  # Specific
            r_before = c.r_orf[i_c]
            v_before = c.v_orf[i_c]
            dv = 1e-2
            v_deploy = v.RVW_ChipSat_SPREAD[1]
            self.r_orf = [r_before.copy() for _ in range(self.n)]
            self.v_orf = [v_before + np.array([0, 0, v_deploy]) + np.random.uniform(-dv, dv, 3) for _ in range(self.n)]
            self.w_orf = [v.spread('w', name=self.name) for _ in range(self.n)]
