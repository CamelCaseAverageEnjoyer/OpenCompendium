from cosmetic import my_print

def get_types_dict():
    """Костыль на тип данных параметров при загрузке и сохранении.
    Используется в config.load_params и config.save_params"""
    return {'CUBESAT_AMOUNT': 'int32', 'CHIPSAT_AMOUNT': 'int32', 'START_NAVIGATION_N': 'int32',
            'GAIN_MODEL_C_N': 'int32', 'GAIN_MODEL_F_N': 'int32', 'CUBESAT_MODEL_N': 'int32',
            'CHIPSAT_MODEL_N': 'int32', 'DEPLOYMENT_N': 'int32'}

class Variables:
    def get_saving_params(self):
        """Функция возвращает набор параметров для записи в файл
        Должно быть согласовано с: self.set_saving_params(), data/config_choose.csv"""
        q = " ".join([str(i) for i in self.KALMAN_COEF['q']])
        p = " ".join([str(i) for i in self.KALMAN_COEF['p']])
        rvw_cubesat = " ".join([str(i) for i in self.RVW_CubeSat_SPREAD])
        rvw_chipsat = " ".join([str(i) for i in self.RVW_ChipSat_SPREAD])
        return [self.DESCRIPTION, self.dT, self.TIME, self.DISTORTION,
                self.CUBESAT_AMOUNT, self.CHIPSAT_AMOUNT, self.DYNAMIC_MODEL['aero drag'], self.DYNAMIC_MODEL['j2'],
                self.NAVIGATION_ANGLES, self.START_NAVIGATION_N, self.GAIN_MODEL_C_N, self.GAIN_MODEL_F_N,
                self.IF_NAVIGATION, self.CUBESAT_MODEL_N, self.CHIPSAT_MODEL_N, q, p, self.KALMAN_COEF['r'],
                rvw_cubesat, rvw_chipsat, self.DEPLOYMENT_N]

    def set_saving_params(self, params):
        """Функция принимает набор параметров из файла
        Должно быть согласовано с: self.get_saving_params(), data/config_choose.csv"""
        self.DESCRIPTION, self.dT, self.TIME, self.DISTORTION, self.CUBESAT_AMOUNT, self.CHIPSAT_AMOUNT, \
            aero, j2, self.NAVIGATION_ANGLES, self.START_NAVIGATION_N, self.GAIN_MODEL_C_N, self.GAIN_MODEL_F_N, \
            self.IF_NAVIGATION, self.CUBESAT_MODEL_N, self.CHIPSAT_MODEL_N, q, p, r, rvw_cubesat, rvw_chipsat, \
            self.DEPLOYMENT_N = params
        self.DYNAMIC_MODEL['aero drag'] = aero
        self.DYNAMIC_MODEL['j2'] = j2
        self.KALMAN_COEF['q'] = [float(i) for i in q.split()]
        self.KALMAN_COEF['p'] = [float(i) for i in p.split()]
        self.KALMAN_COEF['r'] = float(r)
        self.RVW_CubeSat_SPREAD = [float(i) for i in rvw_cubesat.split()]
        self.RVW_ChipSat_SPREAD = [float(i) for i in rvw_chipsat.split()]

        self.init_choice_params()

    def load_params(self, i: int = 0):
        """Подгрузка параметров из файла data/config_choose.csv"""
        import pandas as pd
        from os.path import isfile
        if isfile(self.path_config_data):
            self.config_choose = pd.read_csv(self.path_config_data, sep=";")
            self.config_choose = self.config_choose.astype(get_types_dict())

            self.set_saving_params(self.config_choose.iloc[i, :].to_list())
            self.init_choice_params()
            my_print(f"Загружены параметры: {self.DESCRIPTION}", color='m', if_print=self.IF_ANY_PRINT)

    def save_params(self, add_now_params: bool = True):
        """Сохранение параметров в файл data/config_choose.csv"""
        self.config_choose = self.config_choose.reset_index(drop=True)
        if add_now_params:  # Нужно для специфики self.remove_params()
            self.config_choose.loc[len(self.config_choose), :] = self.get_saving_params()
        self.config_choose = self.config_choose.astype(get_types_dict())  # Установление типов данных
        self.config_choose.to_csv(self.path_config_data, sep=";")
        with open(self.path_config_data, 'r') as f:  # Костыль на то, чтобы убрать ";"
            s = f.read()
        with open(self.path_config_data, 'w') as f:
            f.write(s[1:])
        my_print(f"Параметры сохранены!")
        self.load_params(i=len(self.config_choose)-1)

    def remove_params(self, i: int = 0):
        """Функция убирает строку параметров, сохраняет в файл"""
        tmp = self.config_choose.loc[i, 'DESCRIPTION']
        self.config_choose = self.config_choose.drop(i)
        my_print(f"Удалены параметры {tmp}", color="r")
        self.save_params(add_now_params=False)

    def __init__(self):
        from spacecrafts import Anchor
        from my_math import deg2rad
        import numpy as np

        # >>>>>>>>>>>> Вручную настраиваемые параметры <<<<<<<<<<<<
        self.path_sources = "kiamformation/data/"
        self.path_config_data = self.path_sources + "config_choose.csv"
        self.DESCRIPTION = "По умолчанию"

        self.dT = 1.
        self.TIME = 1e4
        self.CUBESAT_AMOUNT = 1
        self.CHIPSAT_AMOUNT = 1
        self.DYNAMIC_MODEL = {'aero drag': False,
                              'j2': False}
        self.NAVIGATION_ANGLES = False  # Содержит ли искомый вектор состояния кватернионы и угловые скорости
        self.IF_NAVIGATION = True

        self.RVW_CubeSat_SPREAD = [1e2, 1e-1, 1e-4]  # r (м), v (м/с), ω (рад/с)
        self.RVW_ChipSat_SPREAD = [1e2, 1e-1, 1e-4]
        self.KALMAN_COEF = {'q': [1e-15]*2, 'p': [1e-8]*4, 'r': 1e-1}
        self.SHAMANISM = {'KalmanSpinLimit': [True, 1e-2],  # Ограничение скорости вращения в прогнозе фильтра Калмана
                          'ClohessyWiltshireC1=0': True,  # Траектории без дрейфа (зануление C1, даже при аэродинамике)
                          'KalmanVelocityLimit': [False, 1e3],
                          'KalmanPositionLimit': [False, 1e4]}

        self.DISTORTION = 0  # Искривление диаграммы направленности
        self.START_NAVIGATION_TOLERANCE = 0.9


        # >>>>>>>>>>>> Параметры с выбором <<<<<<<<<<<<
        self.START_NAVIGATION_N = 1
        self.GAIN_MODEL_C_N = 0
        self.GAIN_MODEL_F_N = 0
        self.SOLVER_N = 0
        self.CUBESAT_MODEL_N = 0
        self.CHIPSAT_MODEL_N = 0
        self.ATMOSPHERE_MODEL_N = 0
        self.DEPLOYMENT_N = 0

        self.dTs = ["0.1", "1.0", "10.0", "30.0", "100.0"]
        self.Ts = ["100.0", "1000.0", "5000.0", "10000.0", "100000.0"]
        self.CUBESAT_MODELS = ['1U', '1.5U', '2U', '3U', '6U', '12U']
        self.CHIPSAT_MODELS = ['KickSat', 'Трисат']
        self.DEPLOYMENTS = ['No', 'Special']
        self.GAIN_MODES = ['isotropic', '1 antenna', '2 antennas', '3 antennas']
        self.N_ANTENNAS = {'isotropic': 1, '1 antenna': 1, '2 antennas': 2, '3 antennas': 3}
        self.NAVIGATIONS = ['perfect', 'near', 'random']
        self.SOLVERS = ['rk4 hkw', 'kiamastro']
        self.OPERATING_MODES = ['free_flying', 'swarm_stabilize', 'lost']  # Пока что нигде не используется
        self.OPERATING_MODES_CHANGE = ['const', 'while_sun_visible']
        self.MY_COLORMAPS = ['cool', 'winter', 'summer', 'spring', 'gray', 'bone', 'autumn']
        self.ATMOSPHERE_MODELS = ['NASA', 'ПНБО', 'COESA62', 'COESA76']
        self.MY_COLORS = ['violet', 'forestgreen', 'cornflowerblue', 'peru', 'teal', 'blueviolet', 'deeppink',
                          'darksalmon', 'magenta', 'maroon', 'orchid', 'purple', 'wheat', 'tan', 'steelblue',
                          'forestgreen', 'aqua', 'blue', 'beige', 'bisque', 'indigo', 'navy', 'deepskyblue', 'gold',
                          'aquamarine', 'indigo', 'olivedrab', 'slategray', 'pink', 'salmon', 'steelblue']

        self.START_NAVIGATION, self.GAIN_MODEL_C, self.GAIN_MODEL_F, self.SOLVER, self.CUBESAT_MODEL, \
            self.CHIPSAT_MODEL, self.ATMOSPHERE_MODEL, self.N_ANTENNA_C, self.N_ANTENNA_F, self.DEPLOYMENT = [None] * 10
        self.init_choice_params()


        # >>>>>>>>>>>> Параметры отображения <<<<<<<<<<<<
        self.IF_TALK = False
        self.IF_ANY_PRINT = True
        self.IF_TEST_PRINT = True
        self.IF_ANY_SHOW = False  # а ты к чему относишься?
        self.RELATIVE_SIDES = False
        self.NO_LINE_FLAG = -10
        self.EARTH_FILE_NAME = ["earth1.jpg", "earth2.jpg", "earth3.webp"][1]
        self.LANGUAGE = ['рус', 'eng'][1]

        # >>>>>>>>>>>> Константы <<<<<<<<<<<<
        self.ECCENTRICITY = 0.0
        self.INCLINATION = deg2rad(0)  # В градусах
        self.EARTH_RADIUS = 6371e3  # kiam.units('earth')['DistUnit'] * 1e3
        self.HEIGHT = 500e3
        self.ORBIT_RADIUS = self.EARTH_RADIUS + self.HEIGHT

        # Параметры орбиты
        self.APOGEE = self.ORBIT_RADIUS  # Апогей
        self.PERIGEE = self.ORBIT_RADIUS * (1 - self.ECCENTRICITY)/(1 + self.ECCENTRICITY)  # Перигей
        self.P = self.APOGEE * (1 - self.ECCENTRICITY**2)  # Фокальный параметр
        self.MU = 5.972e24 * 6.67408e-11  # Гравитационный параметр
        self.W_ORB = np.sqrt(self.MU / self.ORBIT_RADIUS ** 3)
        self.W_ORB_VEC_IRF = self.W_ORB * np.array([0, -np.sin(self.INCLINATION), np.cos(self.INCLINATION)])
        self.V_ORB = np.sqrt(self.MU / self.ORBIT_RADIUS)
        self.J2 = 1.082 * 1e-3
        self.RHO = 0.

        self.MY_SEC_IN_TURN = 2 * np.pi / self.W_ORB
        TimeUnit = 0.009322440916154166  # kiam.units('earth')['TimeUnit']
        self.SEC_IN_TURN = 24*3600*TimeUnit*2*np.pi
        self.SEC_IN_RAD = 24*3600*TimeUnit


        # >>>>>>>>>>>> Изменяемые параметры по ходу работы кода <<<<<<<<<<<<
        self.MEASURES_VECTOR = None
        self.MEASURES_VECTOR_NOTES = None

        # >>>>>>>>>>>> Параметры для тестов <<<<<<<<<<<<
        self.ANCHOR = Anchor(v=self)

        # >>>>>>>>>>>> Ты сам выбрал этот путь, никто тебя не заставлял! <<<<<<<<<<<<
        self.config_choose = None
        self.load_params()

        # >>>>>>>>>>>> Специальные начальные условия <<<<<<<<<<<<
        self.specific_initial = {'CubeSat r': np.zeros(3), 'CubeSat v': np.zeros(3),
                                 'CubeSat w': np.zeros(3),
                                 'CubeSat q': np.quaternion(1, -1, -1, -1) / 2,
                                 'CubeSat q-vec': np.array([-1, -1, -1]) / 2,
                                 'ChipSat r': np.array([100, 0, 0]), 'ChipSat v': np.array([0, 0.1, 0.1]),
                                 'ChipSat w': np.array([0, 0.0001, 0]),
                                 'ChipSat q': np.quaternion(1, -1, -1, -1) / 2,
                                 'ChipSat q-vec': np.array([-1, -1, -1]) / 2,
                                 'ChipSat dr': np.array([10, 10, 10]),
                                 'ChipSat dv': np.array([0.01, 0.01, 0.01])}

    def test_mode(self):
        self.IF_TALK = False
        self.IF_ANY_PRINT = False
        self.IF_NAVIGATION = False

    def init_choice_params(self):
        self.START_NAVIGATION = self.NAVIGATIONS[self.START_NAVIGATION_N]
        self.GAIN_MODEL_C = self.GAIN_MODES[self.GAIN_MODEL_C_N]
        self.GAIN_MODEL_F = self.GAIN_MODES[self.GAIN_MODEL_F_N]
        self.SOLVER = self.SOLVERS[self.SOLVER_N]
        self.CUBESAT_MODEL = self.CUBESAT_MODELS[self.CUBESAT_MODEL_N]
        self.CHIPSAT_MODEL = self.CHIPSAT_MODELS[self.CHIPSAT_MODEL_N]
        self.ATMOSPHERE_MODEL = self.ATMOSPHERE_MODELS[self.ATMOSPHERE_MODEL_N]
        self.N_ANTENNA_C = self.N_ANTENNAS[self.GAIN_MODEL_C]
        self.N_ANTENNA_F = self.N_ANTENNAS[self.GAIN_MODEL_F]
        self.DEPLOYMENT = self.DEPLOYMENTS[self.DEPLOYMENT_N]

    def spread(self, param: str, name: str):
        import numpy as np
        _i = 'rvw'.index(param)
        if name == "FemtoSat":
            return np.random.uniform(-self.RVW_ChipSat_SPREAD[_i], self.RVW_ChipSat_SPREAD[_i], 3)
        if name == "CubeSat":
            return np.random.uniform(-self.RVW_CubeSat_SPREAD[_i], self.RVW_CubeSat_SPREAD[_i], 3)


class Objects:
    def __init__(self, v: Variables):
        """Класс объединяет следующие другие классы: CubeSat, FemtoSat, PhysicModel"""
        # Классы
        self.v, self.a, self.c, self.f, self.p = v, None, None, None, None
        self.init_classes()

    def reset(self, config_choose_n):
        self.v.load_params(i=config_choose_n)
        self.init_classes()

    def init_classes(self):
        from dynamics import PhysicModel
        from spacecrafts import CubeSat, FemtoSat
        self.a = self.v.ANCHOR
        self.c = CubeSat(v=self.v)
        self.f = FemtoSat(v=self.v, c=self.c)
        self.p = PhysicModel(c=self.c, f=self.f, a=self.a, v=self.v)

    def time_message(self, t):
        import numpy as np
        return f"Оборотов вокруг Земли: {round(t / (2 * np.pi / self.v.W_ORB), 2)}  ({round(t / (3600 * 24), 2)} дней)"

    def integrate(self, t: float, animate: bool = False) -> None:
        from cosmetic import real_workload_time, my_print
        from my_plot import plot_all
        from datetime import datetime

        my_print(self.time_message(t), color='b', if_print=self.v.IF_ANY_PRINT)
        n = int(t // self.v.dT)
        flag = [0., 0.]
        frames = []
        time_begin = datetime.now()
        for i in range(n):
            # Отображение в вывод
            if i == 1 and self.v.IF_ANY_PRINT:
                # Вывод основных параметров
                tmp = ", ориентации" if self.v.NAVIGATION_ANGLES else ""
                my_print(f"Диаграмма антенн кубсата: {self.c.gain_mode}\n"
                         f"Диаграмма антенн фемтосатов: {self.f.gain_mode}\n"
                         f"Учёт аэродинамики: {self.v.DYNAMIC_MODEL['aero drag']}\n"
                         f"Применяется фильтр Калмана для поправки: положений, скоростей{tmp}\n", color='c')
                my_print(f"Вариант отделения дочерних КА: {self.v.DEPLOYMENT}", color='m')
                my_print(f"Внимание: IF_NAVIGATION={self.v.IF_NAVIGATION}! ", color='m',
                         if_print=not self.v.IF_NAVIGATION)
                my_print(f"Шаг моделирования: {self.v.dT}")
            if i / n > (flag[0] + 0.1):
                flag[0] += 0.1
                per = int(10 * i / n)
                my_print(f"{10 * per}% [{'#' * per + ' ' * (10 - per)}]" +
                         real_workload_time(n=per, n_total=10, time_begin=time_begin,
                                            time_now=datetime.now()), color='m', if_print=self.v.IF_ANY_PRINT)

            # Отображение в анимацию
            if animate and i / n > (flag[1] + 0.01):
                flag[1] += 0.01
                frames.append(plot_all(self, save=True, count=int(flag[1] // 0.01)))

            # Шаг по времени
            self.p.time_step()

def init():
    return Objects(v=Variables())
