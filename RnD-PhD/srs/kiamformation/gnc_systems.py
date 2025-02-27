import numpy as np
import control

from spacecrafts import *
from H_matrix import *
from symbolic import *
from cosmetic import *

# >>>>>>>>>>>> Guidance <<<<<<<<<<<<
def guidance(c: CubeSat, f: FemtoSat, v: Variables, earth_turn: float) -> None:
    """Функция обновляет для объектов CubeSat и FemtoSat параметры b_env"""
    for obj in [c, f]:
        for i in range(obj.n):
            pass
            # if obj.operating_modes[i] == v.OPERATING_MODES_CHANGE[1]:  # Отсутствие аккумулятора на чипсате
            #     if earth_turn % 1 < 0.5 and obj.operating_mode[i] == v.OPERATING_MODES[-1]:
            #         obj.operating_mode[i] = v.OPERATING_MODES[0]
            #     if earth_turn % 1 > 0.5 and obj.operating_mode[i] != v.OPERATING_MODES[-1]:
            #         obj.operating_mode[i] = v.OPERATING_MODES[-1]

# >>>>>>>>>>>> Navigation <<<<<<<<<<<<
class KalmanFilter:
    """Оцениваемые параметры: ['r orf', 'q-3 irf', 'v orf', 'w brf']; согласовано с spacecrafts.py"""
    def __init__(self, f: FemtoSat, c: CubeSat, p: any):
        # Общие параметры
        self.f = f  # Дочерние КА
        self.c = c  # Материнские КА
        self.p = p  # Динамическая модель
        self.v = p.v

        self.estimation_params = self.params_dict2vec(d=f.apriori_params, separate_spacecraft=True)
        self.j = len(self.estimation_params[0])  # Вектор состояния 1 чипсата
        self.estimation_params_d = self.params_vec2dict(self.estimation_params)
        self.STM = None
        self.observability_gramian = None

        # Матрицы фильтра в начальный момент времени
        if not self.v.NAVIGATION_ANGLES:  # Вектор состояния содержит только положение и скорость
            self.D = np.vstack([np.zeros((3, 3)), np.eye(3)])
            self.P = [np.diag([self.v.KALMAN_COEF['p'][0]] * 3 + [self.v.KALMAN_COEF['p'][1]] * 3) for _ in range(f.n)]
            self.Q = np.diag([self.v.KALMAN_COEF['q'][0]] * 3)
        else:  # Вектор состояния содержит ещё и угловые переменные
            self.D = np.vstack([np.zeros((6, 6)), np.eye(6)])
            self.P = [np.diag([self.v.KALMAN_COEF['p'][0]] * 3 + [self.v.KALMAN_COEF['p'][1]] * 3 +
                              [self.v.KALMAN_COEF['p'][2]] * 3 + [self.v.KALMAN_COEF['p'][3]] * 3) for _ in range(f.n)]
            self.Q = np.diag([self.v.KALMAN_COEF['q'][0]] * 3 + [self.v.KALMAN_COEF['q'][1]] * 3)
        self.Phi = self.get_Phi(w=None, w0=None)

        # Расширешние на учёт несколько аппаратов в фильтре
        self.D = block_diag(*[self.D for _ in range(self.f.n)])
        self.P = block_diag(*[self.P[i] for i in range(self.f.n)])
        self.Q = block_diag(*[self.Q for _ in range(self.f.n)])

        # Вывод
        if self.v.IF_TEST_PRINT:
            my_print(f"Матрицы Ф:{self.Phi.shape}, Q:{self.Q.shape}, P:{self.P.shape}, D:{self.D.shape}", color='g')

    def get_Phi_1(self, i: int, w=None, w0=None, q=None):
        from dynamics import get_matrices
        w0 = self.v.W_ORB if w0 is None else w0
        q = self.f.q if q is None else q
        if not self.v.NAVIGATION_ANGLES:  # Оценка орбитального движения
            return np.array([[0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, -2 * w0],
                             [0, - w0 ** 2, 0, 0, 0, 0],
                             [0, 0, 3 * w0 ** 2, 2 * w0, 0, 0]]) * self.v.dT + np.eye(self.j)
        else:  # Оценка орбитального и углового движения
            J = self.f.J
            Wj = inv(J) @ (-get_antisymmetric_matrix(w) @ J + get_antisymmetric_matrix(J @ w))
            Ww = get_antisymmetric_matrix(w)

            U, S, A, R_orb = get_matrices(v=self.v, t=self.p.t, obj=self.f, n=i)
            R = quart2dcm(vec2quat(*np.array(self.estimation_params_d['q-3 irf']))) @ R_orb
            eta = R / np.linalg.norm(R)
            Wr = get_antisymmetric_matrix(eta)
            Wq = inv(self.f.J) @ (6*self.v.W_ORB**2 * (Wr @ J @ Wr - get_antisymmetric_matrix(J @ eta) @ Wr))

            if self.v.DYNAMIC_MODEL['aero drag']:
                q_x, q_y, q_z = q
                t = self.p.t
                w_0 = self.v.W_ORB

                s1 = (2*q_x**2/sqrt(-q_x**2 - q_y**2 - q_z**2 + 1) - 2*sqrt(-q_x**2 - q_y**2 - q_z**2 + 1))*cos(t*w_0) - (2*q_x*q_y/sqrt(-q_x**2 - q_y**2 - q_z**2 + 1) + 2*q_z)*sin(t*w_0)
                s2 = -(2*q_y**2/sqrt(-q_x**2 - q_y**2 - q_z**2 + 1) - 2*sqrt(-q_x**2 - q_y**2 - q_z**2 + 1))*sin(t*w_0) + (2*q_x*q_y/sqrt(-q_x**2 - q_y**2 - q_z**2 + 1) + 2*q_z)*cos(t*w_0)
                s3 = -(2*q_x + 2*q_y*q_z/sqrt(-q_x**2 - q_y**2 - q_z**2 + 1))*sin(t*w_0) + (2*q_x*q_z/sqrt(-q_x**2 - q_y**2 - q_z**2 + 1) + 2*q_y)*cos(t*w_0)

                c = self.f.get_blown_surface(cos_alpha=1) * self.v.RHO / self.f.mass
                v2 = self.v.V_ORB**2
                s1, s2, s3 = c*s1*v2, c*s2*v2, c*s3*v2
            else:
                s1, s2, s3 = 0, 0, 0

            O = np.zeros((3, 3))
            E = np.eye(3)
            A = np.array([[0, 0, 0],
                          [0, -w0**2, 0],
                          [0, 0, 3*w0**2]])
            B = np.array([[s1, s2, s3],
                          [0, 0, 0],
                          [0, 0, 0]])
            C = np.array([[0, 0, -2*w0],
                          [0, 0, 0],
                          [2*w0, 0, 0]])
            F = bmat([[O, O, E, O],
                      [O, -Ww, O, E/2],
                      [A, B, C, O],
                      [O, Wq, O, Wj]])

            return F * self.v.dT + np.eye(self.j)

    def get_Phi(self, w=None, w0=None, q=None):
        w0 = self.v.W_ORB if w0 is None else w0
        w = [self.estimation_params_d['w brf'][i] for i in range(self.f.n)] if w is None else w
        q = [self.estimation_params_d['q-3 irf'][i] for i in range(self.f.n)] if q is None else q
        return block_diag(*[self.get_Phi_1(w=w[i], w0=w0, i=i, q=q[i]) for i in range(self.f.n)])

    def params_vec2dict(self, params: list = None, j: int = None, separate_spacecraft: bool = True):
        p = self.estimation_params if params is None else params
        j = self.j if j is None else j
        if self.v.NAVIGATION_ANGLES:
            if separate_spacecraft:
                r_orf = [p[i][0: 3] for i in range(self.f.n)]
                q_irf = [p[i][3: 6] for i in range(self.f.n)]
                v_orf = [p[i][6: 9] for i in range(self.f.n)]
                w_orf = [p[i][9: 12] for i in range(self.f.n)]
            else:
                r_orf = [p[i*j + 0: i*j + 3] for i in range(self.f.n)]
                q_irf = [p[i*j + 3: i*j + 6] for i in range(self.f.n)]
                v_orf = [p[i*j + 6: i*j + 9] for i in range(self.f.n)]
                w_orf = [p[i*j + 9: i*j + 12] for i in range(self.f.n)]
        else:
            if separate_spacecraft:
                r_orf = [p[i][0: 3] for i in range(self.f.n)]
                v_orf = [p[i][3: 6] for i in range(self.f.n)]
            else:
                r_orf = [p[i*j + 0: i*j + 3] for i in range(self.f.n)]
                v_orf = [p[i*j + 3: i*j + 6] for i in range(self.f.n)]
            q_irf, w_orf = [[None for _ in range(self.f.n)] for _ in range(2)]
        return {'r orf': r_orf, 'v orf': v_orf, 'w brf': w_orf, 'q-3 irf': q_irf}

    def params_dict2vec(self, d: dict, separate_spacecraft: bool = True):
        variables = ['r orf', 'q-3 irf', 'v orf', 'w brf'] if self.v.NAVIGATION_ANGLES else ['r orf', 'v orf']
        if separate_spacecraft:
            return [np.array([d[v][i][j] for v in variables for j in range(3)]) for i in range(self.f.n)]
        else:
            return np.array([d[v][i][j] for i in range(self.f.n) for v in variables for j in range(3)])

    def get_estimation(self, i_f: int, v: str):
        d = self.params_vec2dict()
        return d[v][i_f]

    def calc(self, if_correction: bool) -> None:
        from primary_info import measure_antennas_power
        from dynamics import rk4_translate, rk4_attitude

        # >>>>>>>>>>>> Предварительный расчёт <<<<<<<<<<<<
        if True:
            f = self.f
            c = self.c
            v = self.v
            p = self.p
            j = self.j
            c_take_len = len(get_gain(v=v, obj=c, r=np.ones(3), if_take=True))
            c_send_len = len(get_gain(v=v, obj=c, r=np.ones(3), if_send=True))
            f_take_len = len(get_gain(v=v, obj=f, r=np.ones(3), if_take=True))
            f_send_len = len(get_gain(v=v, obj=f, r=np.ones(3), if_send=True))
            z_len = int((f.n * c.n * (c_take_len*f_send_len + c_send_len*f_take_len) +
                         f.n * (f.n - 1) * f_take_len*f_send_len) // 2)
            my_print(f"Вектор измерений 2*z_len: {z_len}     =     {f.n}*{c.n}*({c_take_len}*{f_send_len}+"
                     f"{c_send_len}*{f_take_len}) + {f.n}({f.n}-1){f_take_len}*{f_send_len}",
                     color='r', if_print=p.iter == 1 and v.IF_TEST_PRINT)

        # >>>>>>>>>>>> Этап экстраполяции <<<<<<<<<<<<
        d = self.params_vec2dict()

        # Моделирование орбитального движения на dT -> вектор состояния x_m
        rv_m = [rk4_translate(v_=v, obj=f, i=i, r=d['r orf'][i], v=d['v orf'][i]) for i in range(f.n)]
        qw_m = [rk4_attitude(v_=v, obj=f, i=i, t=self.p.t, q=d['q-3 irf'][i], w=d['w brf'][i]) for i in range(f.n)]
        x_m = self.params_dict2vec(d={'r orf': [rv_m[i][0] for i in range(f.n)],
                                      'v orf': [rv_m[i][1] for i in range(f.n)],
                                      'q-3 irf': [qw_m[i][0] for i in range(f.n)],
                                      'w brf': [qw_m[i][1] for i in range(f.n)]}, separate_spacecraft=False)
        d = self.params_vec2dict(params=x_m, separate_spacecraft=False)
        self.estimation_params_d = d

        # Измерения с поправкой на угловой коэффициент усиления G (signal_rate)
        z_ = v.MEASURES_VECTOR
        notes1 = v.MEASURES_VECTOR_NOTES

        # Измерения согласно модели
        z_model, _, notes3 = measure_antennas_power(c=c, f=f, v=v, p=p, j=j, estimated_params=x_m)
        tmp = np.abs(z_model - z_)
        p.record.loc[p.iter, f'ZModel&RealDifference'] = tmp.mean()
        p.record.loc[p.iter, f'ZModel&RealDifference min'] = tmp.min()
        p.record.loc[p.iter, f'ZModel&RealDifference max'] = tmp.max()
        p.record.loc[p.iter, f'ZModel&RealDifference N'] = len(z_model)
        p.record.loc[p.iter, f'ZReal N'] = len(z_)
        p.record.loc[p.iter, f'ZModel N'] = len(z_model)
        for i in range(len(z_model)):
            p.record.loc[p.iter, f'ZModel&RealDifference {i}'] = tmp[i]
            p.record.loc[p.iter, f'ZReal {i}'] = abs(z_[i])
            p.record.loc[p.iter, f'ZModel {i}'] = abs(z_model[i])

        if if_correction:
            # >>>>>>>>>>>> Этап коррекции <<<<<<<<<<<<
            self.Phi = self.get_Phi(w=None, w0=None)
            Q_tilda = self.Phi @ self.D @ self.Q @ self.D.T @ self.Phi.T * v.dT_nav
            P_m = self.Phi @ self.P @ self.Phi.T + Q_tilda
            q_f = d['q-3 irf'] if v.NAVIGATION_ANGLES else [f.q[i].vec for i in range(f.n)]
            q_c = [c.q[i].vec for i in range(c.n)] if v.NAVIGATION_ANGLES else [c.q[i].vec for i in range(c.n)]
            H = h_matrix(t=p.t, v=v, f=f, c=c, r_f=d['r orf'], r_c=c.r_orf, q_f=q_f, q_c=q_c)

            R = np.eye(z_len) * v.KALMAN_COEF['r']
            k_ = P_m @ H.T @ np.linalg.inv(H @ P_m @ H.T + R)
            self.P = (np.eye(j * f.n) - k_ @ H) @ P_m
            P = self.P
            raw_estimation_params = np.array(np.matrix(x_m) + k_ @ (z_ - z_model))[0]

            # Численный расчёт STM (state transition matrix)
            self.STM = self.Phi if self.STM is None else self.Phi @ self.STM
            tmp = self.STM.T @ H.T @ H @ self.STM
            self.observability_gramian = tmp if self.observability_gramian is None else self.observability_gramian + tmp
            _, simgas, _ = np.linalg.svd(self.observability_gramian)
            p.record.loc[p.iter, f'gramian sigma criteria'] = np.min(simgas)/np.max(simgas)

            tmp = control.obsv((self.Phi - np.eye(self.Phi.shape[0])) / self.v.dT_nav, H)
            _, simgas, _ = np.linalg.svd(tmp)
            p.record.loc[p.iter, f'linear rank criteria'] = np.linalg.matrix_rank(tmp)
            p.record.loc[p.iter, f'linear sigma criteria'] = np.min(simgas)/np.max(simgas)
        else:
            raw_estimation_params = x_m

        # >>>>>>>>>>>> Обновление оценки <<<<<<<<<<<<
        for i in range(f.n):
            tmp = raw_estimation_params[(0 + i) * j: (1 + i) * j]
            if v.SHAMANISM["KalmanQuaternionNormalize"] and v.NAVIGATION_ANGLES:
                tmp2 = vec2quat(tmp[3:6])
                tmp[3:6] = tmp2.normalized().vec
            if v.SHAMANISM["KalmanSpinLimit"][0] and \
                    np.linalg.norm(tmp[9:12]) > v.SHAMANISM["KalmanSpinLimit"][1]:
                tmp[9:12] = tmp[9:12] / np.linalg.norm(tmp[9:12]) * v.SHAMANISM["KalmanSpinLimit"][1]
            if v.SHAMANISM["KalmanVelocityLimit"][0] and \
                    np.linalg.norm(tmp[6:9]) > v.SHAMANISM["KalmanVelocityLimit"][1]:
                tmp[6:9] = tmp[6:9] / np.linalg.norm(tmp[6:9]) * v.SHAMANISM["KalmanSpinLimit"][1]
            if v.SHAMANISM["KalmanPositionLimit"][0] and \
                    np.linalg.norm(tmp[0:3]) > v.SHAMANISM["KalmanPositionLimit"][1]:
                tmp[0:3] = tmp[0:3] / np.linalg.norm(tmp[0:3]) * v.SHAMANISM["KalmanPositionLimit"][1]
            self.estimation_params[i] = tmp

        # Запись и отображение
        if p.iter == 1:
            if if_correction and False:
                my_print(f"P-: {P_m.shape}, H.T: {H.T.shape}, H: {H.shape}, R: {R.shape}", color='g')
                my_print(f"x-: {np.matrix(x_m).shape}, K: {k_.shape}, z: {(z_ - z_model).shape}", color='g')
            my_print(f"R-notes: {notes1}", color='y')
            my_print(f"Длина длин: {len(z_)}", color='r', if_print=v.IF_TEST_PRINT)
            my_print(f"M-notes: {notes3}", color='y')
            my_print(f"Длина модельных длин: {len(z_model)}", color='r', if_print=v.IF_TEST_PRINT)
            with open("kiamformation/data/measures_vector_notes_last.txt", "w") as f:
                f.write("# Рассчитано в PyCharm\n# Параметры: {rel} {N_1} {N_2} {i_1} {i_2} {send_len} {take_len}\n")
                f.write(f"# Параметр CUBESAT_AMOUNT {v.CUBESAT_AMOUNT}\n")
                f.write(f"# Параметр CHIPSAT_AMOUNT {v.CHIPSAT_AMOUNT}\n")
                f.write(f"# Параметр NAVIGATION_ANGLES {v.NAVIGATION_ANGLES}\n")
                f.write(f"# Параметр N_ANTENNA_C {v.N_ANTENNA_C}\n")
                f.write(f"# Параметр N_ANTENNA_F {v.N_ANTENNA_F}\n")
                f.write(f"# Параметр MULTI_ANTENNA_TAKE {v.MULTI_ANTENNA_TAKE}\n")
                f.write(f"# Параметр MULTI_ANTENNA_SEND {v.MULTI_ANTENNA_SEND}\n")
                for j in range(z_len):
                    f.write(f"{v.MEASURES_VECTOR_NOTES[j]}\n")
            my_print(f"P: {self.v.KALMAN_COEF['p']}, Q: {self.v.KALMAN_COEF['q']}", color='g')
            my_print(f"estimation_params: {raw_estimation_params.shape}", color='g')


def navigate(k: KalmanFilter, if_correction: bool = True):
    k.calc(if_correction=if_correction)  # Пока что при любом OPERATING_MODES (если все КА включены и не потеряны)

# >>>>>>>>>>>> Control <<<<<<<<<<<<
