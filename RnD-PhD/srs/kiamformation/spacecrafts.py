"""Функции, связанные с архитектурой КА"""
from config import Variables

# >>>>>>>>>>>> Диаграмма направленности антенн связи <<<<<<<<<<<<
def r_a(ind: str):
    if ind not in 'xyz':
        raise ValueError(f"Координата «{ind}» должна быть среди: [x, y, z]")
    return [int(ind == 'x'), int(ind == 'y'), int(ind == 'z')]

def local_dipole(vrs: Variables, r, ind: str = 'x', model='quarter-wave monopole'):
    """Возвращает коэффициент усиления от 1-й антенны
    :param vrs: Объект класса Variables
    :param r: Радиус-вектор между антенной-трансмиттером и антенной-ресивером в ССК
    :param ind: Координата направления антенны-ресивера в ССК
    :param model: """
    from flexmath import setvectype, norm, cos, dot, cross, pi
    from my_math import vec2unit

    r_12 = vec2unit(r)  # + setvectype([0.03, 0.05, 0]) * vrs.DISTORTION  # Искажение диаграммы направленности
    r_antenna_brf = setvectype(r_a(ind))
    
    sin_theta = norm(cross(r_antenna_brf, r_12))
    cos_theta = dot(r_antenna_brf, r_12)
    if model == 'half-wave dipole':
        return cos(pi(r) / 2 * cos_theta) / sin_theta
    if model == 'short dipole':
        return sin_theta**2
    if model == 'quarter-wave monopole':
        return sin_theta**3

def get_gain(vrs: Variables, obj, r, gm=None, return_dir: bool = False):
    """Возвращает вектор коэффициентов усиления от каждой антенны КА
    :param vrs: Объект класса Variables
    :param obj: Переменная класса Apparatus
    :param r: Направление сигнала в ССК КА
    :param return_dir: Возврат только вектора направления антенны в ССК КА (опционально)
    :param gm: Количество и тип антенн (опционально)
    Памятка: GAIN_MODES = ['isotropic', '1 antenna', '2 antennas', '3 antennas']"""
    gm = obj.gain_mode if gm is None else gm
    if gm in vrs.GAIN_MODES and gm != 'isotropic':
        d = ['x', 'xy', 'xyz'][vrs.GAIN_MODES.index(gm) - 1]
        return [local_dipole(vrs, r, i) for i in d] if not return_dir else [r_a(i) for i in d]
    return [1]


# >>>>>>>>>>>> Классы аппаратов <<<<<<<<<<<<
class Apparatus:
    def __init__(self, v: Variables, n: int):
        """Пустой класс КА"""
        from dynamics import get_matrices, o_i
        import numpy as np

        # Общие параметры
        self.name = "No exist"
        self.n = n
        self.mass = 1e10
        self.size = [1., 1., 1.]
        self.c_resist = 0
        self.J = None
        self.gain_mode = 'isotropic'
        self.b_env = None

        # Индивидуальные параметры движения
        self.w_irf = [np.zeros(3) for _ in range(self.n)]
        self.w_orf = [np.zeros(3) for _ in range(self.n)]
        self.w_brf = [np.zeros(3) for _ in range(self.n)]
        self.q = [np.quaternion(1, 0, 0, 0) for _ in range(self.n)]
        self.r_orf = [np.zeros(3) for _ in range(self.n)]
        self.v_orf = [np.zeros(3) for _ in range(self.n)]
        U, _, _, _ = get_matrices(vrs=v, t=0, obj=self, n=0, first_init=True)
        self.r_irf = [o_i(vrs=v, a=self.r_orf[0], U=U, vec_type='r')]
        self.v_irf = [o_i(vrs=v, a=self.v_orf[0], U=U, vec_type='v')]

        # Индивидуальные параметры режимов работы
        self.operating_mode = [v.OPERATING_MODES[0] for _ in range(self.n)]

    def get_blown_surface(self, cos_alpha):
        return abs(self.size[0] * self.size[1] * abs(cos_alpha) * self.c_resist)

    def update_irf_rv(self, v: Variables, t: float = 0):
        from dynamics import o_i, get_matrices
        for i in range(self.n):
            U, _, _, _ = get_matrices(vrs=v, t=t, obj=self, n=i)
            self.r_irf[i] = o_i(vrs=v, a=self.r_orf[i], U=U, vec_type='r')
            self.v_irf[i] = o_i(vrs=v, a=self.v_orf[i], U=U, vec_type='v')

    def update_irf_w(self, v: Variables, t: float = 0, w_irf: list = None, w_orf: list = None, w_brf: list = None):
        from dynamics import o_i, get_matrices
        w_irf = self.w_irf if w_irf is None else w_irf
        w_orf = self.w_orf if w_orf is None else w_orf
        w_brf = self.w_brf if w_brf is None else w_brf
        for i in range(self.n):
            U, _, A, _ = get_matrices(vrs=v, t=t, obj=self, n=i)
            w_irf[i] = o_i(a=w_orf[i], vrs=v, U=U, vec_type='w')
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
        import numpy as np
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
        self.q = [np.quaternion(1, -1, -1, -1) for _ in range(self.n)]

        # СПЕЦИАЛЬНЫЕ НАЧАЛЬНЫЕ УСЛОВИЯ ДЛЯ УДОВЛЕТВОРЕНИЯ ТРЕБОВАНИЯМ СТАТЬИ
        self.r_orf[0], self.v_orf[0], self.w_orf[0], self.q[0] = [v.specific_initial[f"CubeSat {i}"] for i in "rvwq"]

        # Инициализируется автоматически
        self.init_correct_q_v(v=v)
        self.r_irf, self.v_irf, self.w_irf, self.w_irf = [[np.zeros(3) for _ in range(self.n)] for _ in range(4)]
        self.update_irf_rv(v=v, t=0)
        self.update_irf_w(v=v, t=0)

        # Индивидуальные параметры управления
        self.m_self, self.b_env = [[np.zeros(3) for _ in range(self.n)] for _ in range(2)]

        # Прорисовка ножек
        self.legs_x = 0.0085
        self.legs_z = 0.007

class FemtoSat(Apparatus):
    def __init__(self, v: Variables, c: CubeSat):
        """Класс содержит информацию об n фемтосатах.\n
        Все величны представлены в СИ."""
        import numpy as np
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
        self.r_irf, self.v_irf, self.w_irf, self.w_irf_, self.w_brf, self.w_brf_ = \
            [[np.zeros(3) for _ in range(self.n)] for _ in range(6)]
        self.q, self.q_ = [[np.quaternion(*np.random.uniform(-1, 1, 4)) for _ in range(self.n)] for _ in range(2)]

        # СПЕЦИАЛЬНЫЕ НАЧАЛЬНЫЕ УСЛОВИЯ ДЛЯ УДОВЛЕТВОРЕНИЯ ТРЕБОВАНИЯМ СТАТЬИ
        self.r_orf[0], self.v_orf[0], self.w_orf[0], self.q[0] = [v.specific_initial[f"ChipSat {i}"] for i in "rvwq"]

        # Инициализируется автоматически
        self.init_correct_q_v(v=v)
        self.init_correct_q_v(v=v, q=self.q_)
        self.update_irf_rv(v=v, t=0)
        self.update_irf_w(v=v, t=0)
        self.update_irf_w(v=v, t=0, w_irf=self.w_irf_, w_orf=self.w_orf_, w_brf=self.w_brf_)

        # Индивидуальные параметры управления
        self.m_self, self.b_env = [[np.zeros(3) for _ in range(self.n)] for _ in range(2)]
        dr, dv = [v.specific_initial[f"ChipSat d{i}"] for i in "rv"]

        tol = 1 if v.START_NAVIGATION == v.NAVIGATIONS[0] else v.START_NAVIGATION_TOLERANCE
        tol = 0 if v.START_NAVIGATION == v.NAVIGATIONS[2] else tol

        if v.NAVIGATION_ANGLES:
            self.apriori_params = {'r orf': [self.r_orf[i] * tol + v.spread('r', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'v orf': [self.v_orf[i] * tol + v.spread('v', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'w brf': [self.w_brf[i] * tol + self.w_brf_[i] * (1 - tol) for i in range(self.n)],
                                   'q-3 irf': [self.q[i].vec * tol + self.q_[i].vec * (1 - tol) for i in range(self.n)]}
        else:
            '''self.apriori_params = {'r orf': [self.r_orf[i] * tol + v.spread('r', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'v orf': [self.v_orf[i] * tol + v.spread('v', name=self.name) * (1 - tol)
                                             for i in range(self.n)],
                                   'w brf': [self.w_brf[i] for i in range(self.n)],
                                   'q-3 irf': [self.q[i].vec for i in range(self.n)]}'''
            self.apriori_params = {'r orf': [self.r_orf[i] + dr for i in range(self.n)],
                                   'v orf': [self.v_orf[i] + dv for i in range(self.n)],
                                   'w brf': [self.w_brf[i] for i in range(self.n)],
                                   'q-3 irf': [self.q[i].vec for i in range(self.n)]}

    def deploy(self, v: Variables, c: CubeSat, i_c: int) -> None:
        """Функция отделения задаёт начальные условия для дочерних КА из материнских КА
        :param v: объект Variables
        :param c: объект CubeSat
        :param i_c: id-номер материнского КА, от которого отделяются дочерние КА
        :return: {'r orf': ..., 'v orf': ..., 'q-3 irf': ..., 'w irf': ...}, где значения - list of np.ndarray
        """
        import numpy as np

        if v.DEPLOYMENT == v.DEPLOYMENTS[0]:  # Deploy: "No"
            self.r_orf = [v.spread('r', name=self.name) for _ in range(self.n)]
            self.v_orf = [v.spread('v', name=self.name) for _ in range(self.n)]
            self.w_orf = [v.spread('w', name=self.name) for _ in range(self.n)]
        elif v.DEPLOYMENT == v.DEPLOYMENTS[1]:  # Deploy: "Specific"
            r_before = c.r_orf[i_c]
            v_before = c.v_orf[i_c]
            dv = 1e-2
            v_deploy = v.RVW_ChipSat_SPREAD[1]
            self.r_orf = [r_before.copy() for _ in range(self.n)]
            self.v_orf = [v_before + np.array([0, 0, v_deploy]) + np.random.uniform(-dv, dv, 3) for _ in range(self.n)]
            self.w_orf = [v.spread('w', name=self.name) for _ in range(self.n)]
