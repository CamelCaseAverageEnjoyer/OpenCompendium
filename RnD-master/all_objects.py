import numpy as np

from mylibs.construction_functions import *
from mylibs.control_function import *
from mylibs.calculation_functions import *
from mylibs.im_sample import *
from mylibs.numerical_methods import *
from mylibs.plot_functions import *
from mylibs.tiny_functions import *


class AllProblemObjects(object):
    """Класс содержит в себе абсолютно все параметры задачи и некоторые методы
    :param if_impulse_control: Импульсное управление сборочным КА
    :param if_PID_control: Управление на основе ПД-регулятора сборочным КА
    :param if_LQR_control: Управление на основе линейно квадратичного регулятора сборочным КА
    :param if_avoiding: (только для непрерывного управления) Добавка отталкивающих потенциалов для уклонения
    :param N_apparatus: Количество сборочных КА
    :param diff_evolve_vectors: (параметр при выборе дифф эволюции) Количество векторов
    :param diff_evolve_times: (параметр при выборе дифф эволюции) Количество шагов
    :param shooting_amount_repulsion: Количество шагов пристрелки при отталкивании
    :param shooting_amount_impulse: Количество шагов пристрелки импульсного управления
    :param diff_evolve_F: (параметр при выборе дифф эволюции) Вес новых векторов
    :param diff_evolve_chance: (параметр при выборе дифф эволюции) Шанс новых векторов
    :param mu_IPM: Коэффициент метода внутренней точки
    :param T_total: Общее ограничение времени на строительство
    :param T_max: Максимальное время перелёта
    :param T_max_hard_limit: Вот прям вообще максимальное время перелёта, дальше нельзя
    :param freetime: Время неучёта столкновения после отталкивания
    :param dt: Шаг по времени
    :param t_reaction: Время между обнаружением цели и включением управления
    :param time_to_be_busy: Время занятости между перелётами
    :param u_max: Максимальная скорость отталкивания
    :param du_impulse_max: (только для импульсного управления) Максимальная скорость двигательного импульса
    :param w_twist: Угловая скорость закрутки собираемой конструкции по оси Оу ОСК
    :param e_max: (не доделано) Ограничение полной энергии (вращательной и потенциальной угловой) собираемой конструкции
    :param w_max: Ограничение угловой скорости вращения собираемой конструкции
    :param V_max: Ограничение поступательной скорости центра масс собираемой конструкции
    :param R_max: Ограничение допустимое отклонение центра масс собираемой конструкции от центра ОСК
    :param j_max: Ограничение следа матрицы поворота S собираемой конструкции относительно ОСК
    :param a_pid_max: Ограничение ускорения непрерывного управления
    :param is_saving: Флаг записи картинок vedo для дальнейшей анимации
    :param save_rate: Частота записи картинок vedo
    :param coordinate_system: Относительно какой СК будут отображаться 3д-модели
    :param choice: Выбор конструкции
    :param choice_complete: Флаг уже собранной конструкции (не для основного моделирования)
    :param floor: Количество уровней собираемой конструкции (конкретный смысл зависит от типа конструкции)
    :param extrafloor: Количество сверх-уровней собираемой конструкции (чтобы проц сгорел)
    :param if_talk: Флаг устного разговора с пользователем (на свой страх и риск)
    :param if_multiprocessing: Флаг использования нескольких ядер процессора (не для Windows)
    :param if_testing_mode: Флаг отображения экстра-информации
    :param if_any_print: Флаг отображения вообще
    :param Radius_orbit: Радиус круговой орбиты центра ОСК
    :param d_to_grab: Расстояние захвата КА за целевую точку собираемой конструкции
    :param d_crash: Опасное расстояние вокруг элементов собираемой конструкции (диаметры цилиндров стержней)
    :param k_p: (при выборе if_PID_control) Коэффициент ПД-регулятора
    :param k_u: Коэффициент неточности скорости (смотри velocity_spread)
    :param k_av: (при выборе if_avoiding) Коэффициент перед полем
    :param k_ac: Коэффициент постоянного паразитного ускорения (для проверки на неточности)
    :param level_avoid: (при выборе if_avoiding) Степень дистанции до препятствия (1-2 нормально наверное)
    :param s: (для копирования класса AllProblemObjects) Класс собираемой конструкции (переносимые стержни)
    :param c: (для копирования класса AllProblemObjects) Класс собираемой конструкции (непереносимый каркас)
    :param a: (для копирования класса AllProblemObjects) Класс сборочных КА
    :param file_reset: Флаг стирания файла main.txt, где ведётся запись положений объектов и действия КА
    :param method: Метод минимизации функционала, подаваемый в scipy
    :param if_T_in_shooting: Есть ли вариация времени перелёта в пристрелке (дико неустойчивая затея, ужасная даже)
    :param begin_rotation: Оси, относительно которых ИСК превращается в ОСК (повороты 90°)
    """

    def __init__(self,
                 if_impulse_control=False,  # Управление импульсное
                 if_PID_control=False,  # Управление ПД-регулятором
                 if_LQR_control=False,  # Управление ЛКР
                 if_avoiding=False,  # Исскуственное избежание столкновения

                 N_apparatus=1,  # Количество аппаратов
                 diff_evolve_vectors=20,  # Количество проб дифф эволюции
                 diff_evolve_times=5,  # Количество эпох дифф эволюции
                 shooting_amount_repulsion=7,  # Шаги пристрелки отталкивания
                 shooting_amount_impulse=10,  # Шаги пристрелки импульсного управления

                 diff_evolve_F=0.8,  # Гиперпараметр дифф эволюции
                 diff_evolve_chance=0.5,  # Гиперпараметр дифф эволюции
                 mu_IPM=0.001,  # Гиперпараметр дифф эволюции

                 T_total=1e10,  # Необязательное ограничение по времени на строительство
                 T_max=500.,  # Максимальное время перелёта
                 T_max_hard_limit=2000.,  # Максимальное время перелёта при близости нарушении
                 freetime=50.,  # Время неучёта столкновения после отталкивания
                 dt=1.0,  # Шаг по времени
                 t_reaction=10.,  # Время между обнаружением цели и включением управления
                 time_to_be_busy=100.,  # Время занятости между перелётами
                 u_max=0.04,  # Максимальная скорость отталкивания
                 du_impulse_max=0.4,  # Максимальная скорость импульса при импульсном управлении
                 w_twist=0.,
                 e_max=1e10,  # Относительное максимальная допустимое отклонение энергии (иск огр)
                 w_max=0.002,  # Максимально допустимая скорость вращения станции (искуственное ограничение)
                 V_max=0.04,  # Максимально допустимая поступательная скорость станции (искуственное ограничение)
                 R_max=10.,  # Максимально допустимое отклонение станции (искуственное ограничение)
                 j_max=1e9,  # Максимально допустимый след матрицы поворота S (искуственное ограничение)
                 a_pid_max=1e-5,  # Максимальное ускорение при непрерывном управлении

                 is_saving=False,  # Сохранение vedo-изображений
                 save_rate=1,  # Итерации между сохранением vedo-изображений
                 coordinate_system='orbital',  # Система координат vedo-изображения

                 choice='3',  # Тип конструкции
                 choice_complete=False,  # Уже собранная конструкция (для отладки)
                 floor=5,
                 extrafloor=0,

                 if_talk=False,  # Мне было скучно
                 if_multiprocessing=True,  # Многопроцессорность
                 if_testing_mode=False,  # Лишние принтпоинты
                 if_any_print=True,  # Любые принтпоинты

                 Radius_orbit=6800e3,  # Радиус орбиты
                 mu=5.972e24 * 6.67408e-11,  # Гравитационный параметр Земли
                 d_to_grab=0.5,  # Расстояние захвата до цели
                 d_crash=0.1,  # Расстояние соударения до осей стержней

                 k_p=3e-4,  # Коэффициент ПД-регулятора
                 k_u=1e-1,  # Коэффициент разброса скорости
                 k_av=1e-5,  # Коэффициент поля отталкивания
                 k_ac=0.,  # Коэффициент паразитного ускорения
                 level_avoid=2,
                 s=None,  # готовый объект класса Structure
                 c=None,  # готовый объект класса Container
                 a=None,  # готовый объект класса Apparatus
                 file_reset=False,
                 method='trust-const',
                 if_T_in_shooting=False,
                 begin_rotation='xx'):

        # Параметры типа bool
        self.file_name = 'storage/main.txt'
        self.main_numerical_simulation = s is None
        self.survivor = True  # Зафиксирован ли проход "через текстуры" (можно сделать вылет программы)
        self.warning_message = False  # Если где-то проблема, вместо вылета программы я обозначаю её сообщениями
        self.t_flyby = T_max * 0.95  # Время необходимости облёта
        self.if_talk = if_talk
        self.if_multiprocessing = if_multiprocessing
        self.if_testing_mode = if_testing_mode
        self.if_any_print = if_any_print
        self.flag_impulse = True
        self.collision_foo = None

        # Параметры управления
        self.d_crash = d_crash
        self.if_impulse_control = if_impulse_control
        self.if_PID_control = if_PID_control
        self.if_LQR_control = if_LQR_control
        self.if_avoiding = if_avoiding
        self.control = if_impulse_control or if_PID_control or if_LQR_control
        self.diff_evolve_vectors = diff_evolve_vectors
        self.diff_evolve_times = diff_evolve_times
        self.shooting_amount_repulsion = shooting_amount_repulsion
        self.shooting_amount_impulse = shooting_amount_impulse
        self.diff_evolve_F = diff_evolve_F
        self.diff_evolve_chance = diff_evolve_chance
        self.mu_ipm = mu_IPM
        self.k_p = k_p
        self.k_d = kd_from_kp(k_p)
        self.k_u = k_u
        self.k_av = k_av
        self.k_ac = k_ac
        self.level_avoid = level_avoid

        # Параметры времени
        self.T_total = T_total
        self.T_max = T_max
        self.T_max_hard_limit = T_max_hard_limit
        self.freetime = freetime
        self.t = 0.
        self.iter = 0
        self.dt = dt
        self.time_to_be_busy = time_to_be_busy
        self.t_reaction = t_reaction
        self.t_reaction_counter = t_reaction
        self.t_flyby_counter = self.t_flyby

        # Кинематические параметры и ограничения
        self.u_max = u_max
        self.u_min = u_max / 10
        self.du_impulse_max = du_impulse_max
        self.e_max = e_max
        self.w_max = w_max
        self.V_max = V_max
        self.R_max = R_max
        self.j_max = j_max
        self.a_pid_max = a_pid_max
        self.Radius_orbit = Radius_orbit
        self.Re = Radius_orbit
        self.d_to_grab = d_to_grab
        self.mu = mu

        # Случай копирования класса AllProblemObjects
        if s is None:
            self.s, self.c, self.a = get_all_components(choice=choice, complete=choice_complete, n_app=N_apparatus,
                                                        floor=floor, extrafloor=extrafloor)
            if file_reset:
                with open(self.file_name, 'w') as f:
                    f.write(f"ограничения {self.R_max} {self.V_max} {self.j_max} {self.w_max}\n")
                with open('storage/repulsions.txt', 'w') as _:
                    pass
                with open('storage/iteration_docking.txt', 'w') as _:
                    pass
        else:
            self.s, self.c, self.a = (s, c, a)

        # Параметры масс
        self.choice = choice
        self.t_start = np.zeros(self.a.n + 1)  # dt=(t - t_start) -> уравнения ХКУ
        self.M = np.sum(self.s.mass) + np.sum(self.c.mass)
        if self.main_numerical_simulation:
            self.my_print(f"Масса КА m_a = {self.a.mass[0]} кг, Масса конструкции m_ub = "
                          f"{'{:.2f}'.format(np.sum(self.s.mass))}(стержни) + {np.sum(self.c.mass)}(каркас)"
                          f" = {'{:.2f}'.format(self.M)} кг", mode='c')

        #
        self.r_ub = np.zeros(3)
        self.v_ub = np.zeros(3)
        self.taken_beams = np.array([])
        self.taken_beams_p = np.array([])

        # Параметры вращения
        self.w_twist = w_twist
        self.w = np.array([0., w_twist, 0.])
        self.w_diff = 0.
        self.w_hkw = np.sqrt(mu / Radius_orbit ** 3)
        self.w_hkw_vec = np.array([0., 0., self.w_hkw])
        q12 = [[1 / np.sqrt(2) * (begin_rotation[j] in i) for i in ['xyz', 'x', 'y', 'z']] for j in
               range(len(begin_rotation))]
        for i in range(len(q12)):
            self.La = q12[i] if i == 0 else q_dot(q12[i], self.La)
        self.U, self.S, self.A, self.R_e = self.get_matrices(self.La, 0.)
        self.S_0 = self.S
        self.Om = self.U.T @ self.w + self.w_hkw_vec
        self.J, self.r_center = call_inertia(self, [], app_y=0)  # НЕУЧЁТ НЕСКОЛЬКИХ АППАРАТОВ
        self.J_1 = np.linalg.inv(self.J)
        if self.main_numerical_simulation:
            for i in range(self.a.n):
                self.a.r[i] = self.b_o(self.a.target[i])

        # Кинематические параметры
        self.C_R = get_c_hkw(self.r_ub, np.zeros(3), self.w_hkw)
        self.C_r = [get_c_hkw(self.a.r[i], self.a.v[i], self.w_hkw) for i in range(self.a.n)]
        self.v_p = [np.zeros(3) for _ in range(self.a.n)]
        self.dr_p = [np.zeros(3) for _ in range(self.a.n)]
        self.a_orbital = [np.zeros(3) for _ in range(self.a.n)]
        self.A_orbital = np.zeros(3)
        self.a_self = [np.zeros(3) for _ in range(self.a.n)]
        self.a_wrong = np.random.rand(3)
        self.a_wrong = self.a_pid_max * k_ac * self.a_wrong / np.linalg.norm(self.a_wrong)
        self.e = np.zeros(3)  # Угловое ускорение собираемой конструкции

        # Энергетические параметры
        self.U_begin = self.get_potential_energy()
        self.T_begin = self.get_kinetic_energy()
        self.T = 0.
        self.E = 0.
        self.E_max = 0.

        # Прочие параметры
        self.method = method
        self.if_T_in_shooting = if_T_in_shooting
        self.repulsion_counters = [0 for _ in range(self.a.n)]
        self.flag_vision = [False for _ in range(self.a.n)]
        self.a_self_params = [None for _ in range(self.a.n)]

        # Параметры отображения
        self.is_saving = is_saving
        self.save_rate = save_rate
        self.coordinate_system = coordinate_system
        self.frame_counter = 0
        self.line_app_brf = [[] for _ in range(self.a.n)]
        self.line_app_orf = [[] for _ in range(self.a.n)]
        self.line_str_orf = self.r_ub

        # Выбор значений в зависимости от аргументов
        self.cases = dict({'acceleration_control': lambda v: (v if np.linalg.norm(v) < self.a_pid_max else
                                                              v / np.linalg.norm(v) * self.a_pid_max * 0.95),
                           'repulse_vel_control': lambda v: (v if np.linalg.norm(v) < self.u_max else
                                                             v / np.linalg.norm(v) * self.u_max * 0.95)
                           if np.linalg.norm(v) > self.u_min else
                           v / np.linalg.norm(v) * self.u_min * 1.05,
                           'diff_vel_control': lambda a_, cnd: ((a_ if np.linalg.norm(a_) < self.u_max else
                                                                a_ / np.linalg.norm(a_) * self.u_max * 0.95)
                                                                if np.linalg.norm(a_) > self.u_min else
                                                                a_ / np.linalg.norm(a_) * self.u_min * 1.05) if cnd
                           else a_})

    # ----------------------------------------- РАСЧЁТ ПАРАМЕТРОВ
    def get_e_deviation(self) -> float:
        if self.E_max > 1e-4:
            return self.T / self.E_max
        else:
            return self.E

    def get_discrepancy(self, id_app: int, vector: bool = False, r=None) -> any:
        """Возвращает невязку аппарата с целью"""
        tmp = self.a.r[id_app] if r is None else r
        discrepancy = tmp - self.b_o(self.a.target[id_app])
        return discrepancy if vector else np.linalg.norm(discrepancy)

    def get_angular_momentum(self) -> float:
        return np.linalg.norm(self.J @ self.S @ self.w)

    def get_kinetic_energy(self) -> float:
        """Возвращет кинетическую энергию вращения станции"""
        return self.w.T @ self.S.T @ self.J @ self.S @ self.w / 2

    def get_potential_energy(self) -> float:
        tmp = 0
        J_irf = self.b_i(self.J)
        gamma = self.R_e / self.Radius_orbit
        for i in range(3):
            for j in range(3):
                tmp += 3 * J_irf[i][j] * gamma[i] * gamma[j]
        # tmp = 3 * gamma.T @ J_orf @ gamma  # попробовать
        return 1 / 2 * self.mu / self.Radius_orbit ** 3 * (tmp + np.trace(self.J))  # self.mu/self.Radius_orbit + ()

    def get_matrices(self, La=None, t=None) -> tuple:
        """Функция подсчёта матриц поворота из кватернионов; \n
        На вход подаётся кватернион La и скаляр \n
        Заодно считает вектор от центра Земли до центра масс системы в ИСК."""
        La = self.La if (La is None) else La
        La /= np.linalg.norm(La)
        t = self.t if (t is None) else t
        A = quart2dcm(La)
        U = np.array([[0., 1., 0.],
                      [0., 0., 1.],
                      [1., 0., 0.]]) @ \
            np.array([[np.cos(t * self.w_hkw), np.sin(t * self.w_hkw), 0],
                      [-np.sin(t * self.w_hkw), np.cos(t * self.w_hkw), 0],
                      [0, 0, 1]])
        S = A @ U.T
        R_e = U.T @ np.array([0, 0, self.Radius_orbit])
        return U, S, A, R_e

    def get_hkw_acceleration(self, rv) -> np.ndarray:
        return np.array([-2 * self.w_hkw * rv[5],
                         -self.w_hkw ** 2 * rv[1],
                         2 * self.w_hkw * rv[3] + 3 * self.w_hkw ** 2 * rv[2]])

    def get_ext_momentum_rigid_body(self, A, J, R_e):
        return 3 * self.mu * my_cross(A @ R_e, J @ A @ R_e) / self.Radius_orbit ** 5

    def get_masses(self, id_app: int) -> tuple:
        """Нет учёта нескольких аппаратов!"""
        id_beam = self.a.flag_beam[id_app]
        m_beam = 0. if (id_beam is None) else self.s.mass[id_beam]
        m_ub = self.M - m_beam
        m_ss = self.a.mass[id_app] + m_beam
        return m_ss, m_ub

    def get_repulsion(self, id_app: int) -> np.ndarray:
        file = open('storage/repulsions_-1.txt', 'r')
        lcl_counter = self.repulsion_counters[id_app]
        anw = None
        for line in file:
            lst = line.split()
            if int(lst[1]) == id_app:
                if lcl_counter == 0:
                    anw = np.array([float(lst[2 + i]) for i in range(3)])
                lcl_counter -= 1
        file.close()
        return anw

    # ----------------------------------------- РУНГЕ-КУТТЫ 4 ПОРЯДКА
    def rk4_acceleration(self, r, v, a) -> tuple:
        def rv_right_part(rv_: np.ndarray, a_: np.ndarray) -> np.ndarray:
            return np.append(rv_[3:6], a_[0:3])
        rv = np.append(r, v)
        k1 = rv_right_part(rv, a)
        k2 = rv_right_part(rv + k1 * self.dt / 2, a)
        k3 = rv_right_part(rv + k2 * self.dt / 2, a)
        k4 = rv_right_part(rv + k3 * self.dt, a)
        rv = (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
        return rv[0:3] + r, rv[3:6] + v

    def lw_right_part(self, LaOm, t, J) -> np.ndarray:
        """Функция правых частей для угловой скорости; \n
        Используется в методе Рунге-Кутты."""
        La, Om = LaOm[0:4], LaOm[4:7]
        dLa = 1 / 2 * q_dot([0, Om[0], Om[1], Om[2]], La)
        _, _, A, R_e = self.get_matrices(La=La, t=t)
        J1 = np.linalg.inv(J)
        m_external = self.get_ext_momentum_rigid_body(A, J, R_e)
        return np.append(dLa, A.T @ J1 @ (m_external - my_cross(A @ Om, J @ A @ Om)))

    def rk4_w(self, La, Om, J, t) -> tuple:
        """Интегрирование уравнение Эйлера методом Рунге-Кутты 4 порядка; \n
        Используется в виде: \n
        La, Om = rk4_w(La, Om, J, t)."""
        LaOm = np.append(La, Om)  # Запихиваем в один вектор
        k1 = self.lw_right_part(LaOm, t, J)
        k2 = self.lw_right_part(LaOm + k1 * self.dt / 2, t + self.dt / 2, J)
        k3 = self.lw_right_part(LaOm + k2 * self.dt / 2, t + self.dt / 2, J)
        k4 = self.lw_right_part(LaOm + k3 * self.dt, t + self.dt, J)
        LaOm = self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return La + LaOm[0:4], LaOm[4:7] + Om

    # ----------------------------------------- ОБНОВЛЕНИЕ ПЕРЕМЕННЫХ КЛАССА
    def w_update(self) -> None:
        self.w = self.U @ (self.Om - self.w_hkw_vec)

    def om_update(self) -> None:
        self.Om = self.U.T @ self.w + self.w_hkw_vec

    def time_step(self) -> None:
        self.iter += 1
        self.t = self.iter * self.dt

        # Энергия
        self.U = self.get_potential_energy()
        self.T = self.get_kinetic_energy()  # - self.T_begin
        self.E = self.T + self.U  # - self.U_begin - self.T_begin
        self.E_max = max(self.E_max, self.E)

        # Уравнения Эйлера, вращение
        tmp = self.Om.copy()
        self.La, self.Om = self.rk4_w(self.La, self.Om, self.J, self.t)
        self.U, self.S, self.A, self.R_e = self.get_matrices()
        self.w_update()
        self.w_diff = np.linalg.norm(self.w - np.array([0., self.w_twist, 0.]))
        self.e = (self.Om - tmp) / self.dt

        # Движение центра масс собираемой конструкции
        self.r_ub = r_hkw(self.C_R, self.w_hkw, self.t - self.t_start[self.a.n])
        self.v_ub = v_hkw(self.C_R, self.w_hkw, self.t - self.t_start[self.a.n])
        self.A_orbital = self.get_hkw_acceleration(np.append(self.r_ub, self.v_ub))

        # Движения центра масс сборочного КА
        for id_app in self.a.id:
            if self.a.flag_fly[id_app] == 1:
                if self.a.flag_hkw[id_app]:
                    r = r_hkw(self.C_r[id_app], self.w_hkw, self.t - self.t_start[id_app])
                    v = v_hkw(self.C_r[id_app], self.w_hkw, self.t - self.t_start[id_app])
                else:
                    r = self.a.r[id_app]
                    v = self.a.v[id_app]
                    self.a_orbital[id_app] = self.get_hkw_acceleration(np.append(r, v))
                    r, v = self.rk4_acceleration(r, v, self.a_self[id_app] + self.a_orbital[id_app] + self.a_wrong)
            else:
                r = self.b_o(self.a.target_p[id_app])
                v = np.zeros(3)
            self.a_orbital[id_app] = self.get_hkw_acceleration(np.append(r, v))
            self.a.r[id_app] = r
            self.a.v[id_app] = v

            if self.d_crash is not None and self.warning_message and self.main_numerical_simulation \
                    and (self.t - self.t_start[id_app]) > self.freetime and self.survivor and self.if_testing_mode:
                self.survivor = not call_crash(o=self, r_sat=r, R=self.r_ub, S=self.S, taken_beams=self.taken_beams)
                if not self.survivor:
                    self.my_print(get_angry_message(), mode='y')
        if not self.survivor and self.warning_message and self.iter % 200 == 0 and self.if_testing_mode:
            self.my_print(get_angry_message(), mode='r')

    def control_step(self, id_app) -> None:
        """Функция ускорения бортового двигателя / подачи импульса двигателя.
        Вызывается на каждой итерации.
        :param id_app: Id-номер аппарата
        :return: None
        """
        if self.method == 'linear-propulsion' and self.a_self_params[id_app] is not None:
            self.a_self[id_app] = simple_control(self, self.a_self_params[id_app],
                                                 (self.t - self.t_start[id_app]) / self.T_max)
        if self.control and control_condition(o=self, id_app=id_app):
            if self.if_impulse_control:
                impulse_control(o=self, id_app=id_app)
            if self.if_PID_control and self.t_reaction_counter < 0:
                pd_control(o=self, id_app=id_app)
            if self.if_LQR_control and self.t_reaction_counter < 0:
                lqr_control(o=self, id_app=id_app)
            if self.if_avoiding:
                self.a_self[id_app] += avoiding_force(self, id_app)
        if np.linalg.norm(self.a_self[id_app]) > self.a_pid_max:
            self.a_self[id_app] *= self.a_pid_max / np.linalg.norm(self.a_self[id_app])

    def repulse_app_config(self, id_app: int) -> None:
        """Функция, переводящая КА в состояние готовности к прыжку, до момента прыжка
        :param id_app: Id-номер аппарата
        :return: None
        """
        # Алгоритм выбора цели
        if self.a.flag_start[id_app]:  # Если на старте
            id_beam = self.s.call_possible_transport(self.taken_beams)[0]
            self.taken_beams = np.append(self.taken_beams, id_beam)
            self.my_print(f"Аппарат {id_app} забрал стержень {id_beam}", mode="b")
        else:
            if self.a.flag_beam[id_app] is None:  # Возвращение в грузовой контейнер
                id_beam = None
                self.my_print(f"Аппарат {id_app} возвращается на базу", mode="b")
            else:  # Направление к точке закрепления
                id_beam = self.a.flag_beam[id_app]
                self.my_print(f"Аппарат {id_app} летит установливать стержень {id_beam}", mode="b")

        if id_beam is not None:
            r_1 = self.s.r1[id_beam]
        else:
            tmp_id = self.s.call_possible_transport(self.taken_beams)[0]
            r_1 = self.s.r_st[tmp_id] - np.array([self.s.length[tmp_id], 0.0, 0.0])

        self.a.target_p[id_app] = self.a.target[id_app].copy()
        self.a.target[id_app] = r_1
        self.a.flag_beam[id_app] = id_beam
        self.a.flag_start[id_app] = False
        self.a.flag_fly[id_app] = True
        self.a.flag_hkw[id_app] = False if (self.if_PID_control or self.if_LQR_control) else True
        self.a.r_0[id_app] = self.get_discrepancy(id_app)
        self.flag_vision[id_app] = False

    def remove_repulse_app_config(self, id_app: int) -> None:
        """Функция, возвращающая аппарат из состояния готовности
        :param id_app: Id-номер аппарата
        :return: None
        """
        if self.a.flag_beam[id_app] is not None:
            print(f"---------------------------->{self.a.flag_beam[id_app]}")
            id_beam = self.a.flag_beam[id_app]
            self.a.flag_beam[id_app] = None
            self.a.flag_start[id_app] = True
            self.my_print(f"Аппарат {id_app} положил стержень {id_beam} обратно", mode="b")
            self.taken_beams = self.taken_beams_p.copy()

        self.a.target[id_app] = self.a.target_p[id_app].copy()
        self.a.flag_fly[id_app] = False
        self.a.busy_time[id_app] = self.time_to_be_busy * 5
        self.t_reaction_counter = self.t_reaction
        print(f"123123123---------------------------->{self.a.flag_beam[id_app]}")

    def get_repulsion_change_params(self, id_app: int):
        r_ub_orf_p = self.r_ub.copy()
        v_ub_p = self.v_ub.copy()
        r0_crf = np.array(self.a.target_p[id_app])

        J_p, r_center_p = call_inertia(self, self.taken_beams_p, app_y=id_app)
        self.J, self.r_center = call_inertia(self, self.taken_beams, app_n=id_app)
        J_1 = np.linalg.inv(self.J)

        m_ss, m_ub = self.get_masses(id_app)

        R0c = self.r_center - r_center_p  # SRF
        r0c = r0_crf - r_center_p  # SRF

        R0c = -r0c * m_ss / m_ub
        self.r_center = R0c + r_center_p

        r_ub_orf = r_ub_orf_p + self.S.T @ R0c  # ORF
        r_ss_orf = r_ub_orf_p + self.S.T @ r0c  # ORF
        # r_ub_orf = - r_ss_orf * m_ss / m_ub  # ШАМАНСТВО! КОЛДУНСТВО!
        if self.main_numerical_simulation:
            tmp = (m_ss * r_ss_orf + m_ub * r_ub_orf) / (m_ss + m_ub)
            print(f"=====M+M+M+M====>>>> {(m_ss * r0c + m_ub * R0c) / (m_ss + m_ub)}")
            print(f"=====M=M=M=M====>>>> {tmp} ({m_ss}|{m_ub}) ~~~~~~ R={r_ub_orf} / Rp={ self.r_ub}")
            print(f"================>>>> Взятые стержни {self.taken_beams_p} -> {self.taken_beams}")

        return m_ss, m_ub, self.J, J_1, J_p, self.r_center, r_center_p, r_ss_orf, r_ub_orf, r0c, R0c, r_ub_orf_p, v_ub_p

    def repulsion_change_params(self, id_app: int, u0):
        if len(u0.shape) != 1 or len(u0) != 3:
            raise Exception("Неправильный входной вектор скорости!")
        m_ss, m_ub, J, J_1, J_p, r_ub_crf, r_ub_crf_p, r_ss_orf, r_ub_orf, r0c, R0c, r_ub_orf_p, v_ub_p = \
            self.get_repulsion_change_params(id_app)
        u_rot = my_cross(self.w, self.S.T @ r0c)  # ORF
        V_rot = my_cross(self.w, self.S.T @ R0c)  # ORF
        V0 = - u0 * m_ss / m_ub  # BRF
        u = self.S.T @ u0 + v_ub_p + u_rot  # ORF
        V = self.S.T @ V0 + v_ub_p + V_rot  # ORF
        self.w = self.b_o(J_1) @ (
                self.b_o(J_p) @ self.w + (m_ub + m_ss) * my_cross(r_ub_orf_p, v_ub_p) -
                m_ss * my_cross(r_ss_orf, u) - m_ub * my_cross(r_ub_orf, V))  # ORF

        self.om_update()
        self.a.r[id_app] = r_ss_orf
        self.a.v[id_app] = u
        self.C_r[id_app] = get_c_hkw(r_ss_orf, u, self.w_hkw)
        self.C_R = get_c_hkw(r_ub_orf, V, self.w_hkw)
        self.t_start[id_app] = self.t
        self.t_start[self.a.n] = self.t
        self.repulsion_counters[id_app] += 1
        self.r_ub = r_ub_orf.copy()

    def capturing_change_params(self, id_app: int):
        # Параметры пока аппарат летит
        J_p, r_center_p = call_inertia(self, self.taken_beams, app_n=id_app)
        if self.main_numerical_simulation:
            print(f"111111111111111 {self.taken_beams}")
        m_ss, m_ub = self.get_masses(id_app)
        r_ub_p = self.r_ub.copy()
        v_ub_p = self.v_ub.copy()
        r_ss_p = self.a.r[id_app].copy()
        v_ss_p = self.a.v[id_app].copy()

        # Мгновенная посадка и установка стержня
        id_beam = self.a.flag_beam[id_app]
        self.a.target_p[id_app] = self.a.target[id_app].copy()
        if id_beam is not None:
            if np.linalg.norm(np.array(self.s.r1[id_beam]) - np.array(self.a.target[0])) < 1e-2:
                if self.main_numerical_simulation:
                    self.my_print(f"Стержень id:{id_beam} устанавливается", mode="b")
                self.s.flag[id_beam] = np.array([1., 1.])
                self.a.flag_beam[id_app] = None
                self.taken_beams = np.delete(self.taken_beams, np.argmax(self.taken_beams == id_beam))
        else:
            if self.a.target[id_app][0] < -0.6:  # Если "слева" нет промежуточных точек, то окей
                if self.main_numerical_simulation:
                    self.my_print(f'Аппарат id:{id_app} в грузовом отсеке')
                self.a.flag_start[id_app] = True

        # Параметры когда аппарат уже на месте, стержень установлен
        J, r_center = call_inertia(self, self.taken_beams, app_y=id_app)
        if self.main_numerical_simulation:
            print(f"222222222222222 {self.taken_beams}")
        tmp = self.r_ub.copy()
        self.r_ub += self.S.T @ (r_center - r_center_p)  # Спорный момент
        self.v_ub = (v_ub_p * m_ub + v_ss_p * m_ss) / (m_ub + m_ss)
        if self.main_numerical_simulation:
            self.my_print(f"r_ub: {self.r_ub}", mode='r')
            self.my_print(f"v_ub: {self.v_ub}: V={np.linalg.norm(v_ub_p)}, v={np.linalg.norm(v_ss_p)}", mode='r')
        self.w = np.linalg.inv(self.b_o(J)) @ (self.b_o(J_p) @ self.w -
                                               (m_ub + m_ss) * my_cross(self.r_ub, self.v_ub) +
                                               m_ss * my_cross(r_ss_p, v_ss_p) + m_ub * my_cross(r_ub_p, v_ub_p))

        self.om_update()
        self.C_R = get_c_hkw(self.r_ub, self.v_ub, self.w_hkw)
        if self.main_numerical_simulation:
            print(f"--------------------r: {self.r_ub} = {tmp} + ({self.S.T @ r_center} - {self.S.T @ r_center_p})"
                  f"\n--------------------v: {self.v_ub}\n--------------------w: {self.w}")
            print(f"C_R: {self.C_R}")

        # Capture config
        self.taken_beams_p = self.taken_beams.copy()
        self.a.busy_time[id_app] = self.time_to_be_busy
        self.t_reaction_counter = self.t_reaction
        self.t_start[self.a.n] = self.t
        self.a.flag_hkw[id_app] = True
        self.a.flag_fly[id_app] = False
        self.flag_impulse = True

    # ----------------------------------------- ПЕРЕХОДЫ МЕЖДУ СИСТЕМАМИ КООРДИНАТ
    def i_o(self, a, U=None):
        """Инерциальная -> Орбитальная"""
        a_np = np.array(a)
        U = self.U if (U is None) else U
        if len(a_np.shape) == 1:
            return U @ a_np - np.array([0, 0, self.Re])
        if len(a_np.shape) == 2:
            return U @ a_np @ U.T
        raise Exception("Put vector or matrix")

    def o_i(self, a, U=None):
        """Орбитальная -> Инерциальная"""
        a_np = np.array(a)
        U = self.U if (U is None) else U
        if len(a_np.shape) == 1:
            return U.T @ (a_np + np.array([0, 0, self.Re]))
        if len(a_np.shape) == 2:
            return U.T @ a_np @ U
        raise Exception("Put vector or matrix")

    def o_b(self, a, S=None, R=None, r_center=None):
        """Орбитальная -> Связная"""
        a_np = np.array(a)
        S = self.S if (S is None) else S
        R = self.r_ub if (R is None) else R
        r_center = self.r_center if (r_center is None) else r_center
        if len(a_np.shape) == 1:
            return S @ (a_np - R) + r_center
        if len(a_np.shape) == 2:
            return S @ a_np @ S.T
        raise Exception("Put vector or matrix")

    def b_o(self, a, S=None, R=None, r_center=None):
        """Связная -> Орбитальная"""
        a_np = np.array(a)
        S = self.S if (S is None) else S
        R = self.r_ub if (R is None) else R
        r_center = self.r_center if (r_center is None) else r_center
        if len(a_np.shape) == 1:
            return S.T @ (a_np - r_center) + R
        if len(a_np.shape) == 2:
            return S.T @ a_np @ S
        raise Exception("Put vector or matrix")

    def i_b(self, a, U=None, S=None, R=None, r_center=None):
        """Инерциальная -> Связная"""
        a_np = np.array(a)
        U = self.U if (U is None) else U
        S = self.S if (S is None) else S
        R = self.r_ub if (R is None) else R
        r_center = self.r_center if (r_center is None) else r_center
        if len(a_np.shape) == 1:
            return S @ ((U @ a_np - np.array([0, 0, self.Re])) - R) + r_center
        if len(a_np.shape) == 2:
            return S @ U @ a_np @ U.T @ S.T
        raise Exception("Put vector or matrix")

    def b_i(self, a, U=None, S=None, R=None, r_center=None):
        """Связная -> Инерциальная"""
        a_np = np.array(a)
        U = self.U if (U is None) else U
        S = self.S if (S is None) else S
        R = self.r_ub if (R is None) else R
        r_center = self.r_center if (r_center is None) else r_center
        if len(a_np.shape) == 1:
            return U.T @ ((S.T @ (a_np - r_center) + R) + np.array([0, 0, self.Re]))
        if len(a_np.shape) == 2:
            return U.T @ S.T @ a_np @ S @ U
        raise Exception("Put vector or matrix")

    # ----------------------------------------- КОСМЕТИКА
    def file_save(self, txt):
        file = open(self.file_name, 'a')
        file.write(txt + f" {int(self.main_numerical_simulation)}\n")
        file.close()

    def repulsion_save(self, txt):
        file = open('storage/repulsions.txt', 'a')
        file.write(txt + f" {int(self.main_numerical_simulation)}\n")
        file.close()

    def my_print(self, txt, mode=None, test=False):
        """Функция вывода **цветного** текста
        :param txt: выводимый текст
        :param mode: цвет текста {b, g, y, r, c, m}
        :param test: вывод для отладки кода, помечается жёлтым цветом"""
        if (self.if_any_print and not test) or (self.if_testing_mode and test):
            if mode is None and not test:
                print(Style.RESET_ALL + txt)
            if mode == "b":
                print(Fore.BLUE + txt + Style.RESET_ALL)
            if mode == "g":
                print(Fore.GREEN + txt + Style.RESET_ALL)
            if mode == "y" or (test and mode is None):
                print(Fore.YELLOW + txt + Style.RESET_ALL)
            if mode == "r":
                print(Fore.RED + txt + Style.RESET_ALL)
            if mode == "c":
                print(Fore.CYAN + txt + Style.RESET_ALL)
            if mode == "m":
                print(Fore.MAGENTA + txt + Style.RESET_ALL)

    def copy(self):
        slf = AllProblemObjects(choice=self.choice, s=self.s.copy(), c=self.c.copy(), a=self.a.copy())

        slf.main_numerical_simulation = False
        slf.warning_message = False
        slf.t_flyby = self.t_flyby
        slf.if_talk = False
        slf.if_multiprocessing = self.if_multiprocessing
        slf.if_testing_mode = self.if_testing_mode
        slf.if_any_print = self.if_any_print
        slf.flag_impulse = self.flag_impulse
        slf.collision_foo = None

        slf.d_crash = self.d_crash
        slf.if_impulse_control = self.if_impulse_control
        slf.if_PID_control = self.if_PID_control
        slf.if_LQR_control = self.if_LQR_control
        slf.if_avoiding = self.if_avoiding
        slf.control = self.control

        slf.diff_evolve_vectors = self.diff_evolve_vectors
        slf.diff_evolve_times = self.diff_evolve_times
        slf.shooting_amount_repulsion = self.shooting_amount_repulsion
        slf.shooting_amount_impulse = self.shooting_amount_impulse

        slf.diff_evolve_F = self.diff_evolve_F
        slf.diff_evolve_chance = self.diff_evolve_chance
        slf.mu_ipm = self.mu_ipm

        slf.T_total = self.T_total
        slf.T_max = self.T_max
        slf.T_max_hard_limit = self.T_max_hard_limit
        slf.iter = self.iter
        slf.t = self.t
        slf.dt = self.dt
        slf.time_to_be_busy = self.time_to_be_busy
        slf.t_reaction = self.t_reaction
        slf.t_reaction_counter = self.t_reaction_counter
        slf.t_flyby_counter = self.t_flyby
        slf.u_max = self.u_max
        slf.u_min = self.u_min
        slf.du_impulse_max = self.du_impulse_max
        slf.w_max = self.w_max
        slf.V_max = self.V_max
        slf.R_max = self.R_max
        slf.j_max = self.j_max
        slf.a_pid_max = self.a_pid_max

        slf.is_saving = self.is_saving
        slf.save_rate = self.save_rate
        slf.coordinate_system = self.coordinate_system

        slf.Radius_orbit = self.Radius_orbit
        slf.Re = self.Re
        slf.mu = self.mu
        slf.d_to_grab = self.d_to_grab

        slf.k_p = self.k_p
        slf.k_d = self.k_d
        slf.La = copy.deepcopy(self.La)

        slf.t_start = copy.deepcopy(self.t_start)
        slf.M = self.M

        slf.w_hkw = self.w_hkw
        slf.w_hkw_vec = copy.deepcopy(self.w_hkw_vec)
        slf.U = copy.deepcopy(self.U)
        slf.S = copy.deepcopy(self.S)
        slf.A = copy.deepcopy(self.A)
        slf.R_e = copy.deepcopy(self.R_e)

        slf.J = copy.deepcopy(self.J)
        slf.r_center = copy.deepcopy(self.r_center)
        slf.r_ub = copy.deepcopy(self.r_ub)
        slf.v_ub = copy.deepcopy(self.v_ub)
        slf.J_1 = copy.deepcopy(self.J_1)

        slf.taken_beams = copy.deepcopy(self.taken_beams)
        slf.taken_beams_p = copy.deepcopy(self.taken_beams_p)

        slf.C_R = copy.deepcopy(self.C_R)
        slf.C_r = copy.deepcopy(self.C_r)

        slf.v_p = copy.deepcopy(self.v_p)
        slf.dr_p = copy.deepcopy(self.dr_p)
        slf.a_self = copy.deepcopy(self.a_self)
        slf.a_orbital = copy.deepcopy(self.a_orbital)
        slf.A_orbital = copy.deepcopy(self.A_orbital)
        slf.w = copy.deepcopy(self.w)
        slf.w_twist = self.w_twist
        slf.w_diff = self.w_diff
        slf.Om = copy.deepcopy(self.Om)

        slf.E = self.E
        slf.E_max = self.E_max

        slf.method = self.method
        slf.if_T_in_shooting = self.if_T_in_shooting

        return slf
