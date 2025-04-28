"""Комплекс первичной информации"""
from spacecrafts import Apparatus
from config import Variables


def measure_antennas_power(c: Apparatus, f: Apparatus, vrs: Variables, noise: float = None, produce: bool = False,
                           j: int = None, estimated_params=None, p: any = None, t=None):
    """Функция обновляет для объектов CubeSat и FemtoSat параметры calc_dist при produce==True. Иначе:
    :param c: Класс кубсатов
    :param f: Класс чипсатов
    :param vrs: Класс гиперпараметров моделирования
    :param noise:
    :param produce: Флаг, указывающий, надо ли записывать полученные величины в PhysicModel.record
    :param j: Количество параметров на 1 дочерний КА
    :param estimated_params: в одну строку
    :param p: Класс PhysicModel (для флага produce)
    :param t: Время (для символьного вычисления)
    :return: None если produce==True (проведение численного моделирования), иначе список измерений + пометки"""
    # from sympy import Matrix, Rational, Float
    import numpy as np
    from flexmath import setvectype, norm, mean, float2rational
    from my_math import quart2dcm, vec2quat
    from spacecrafts import get_gain

    randy = np.random.uniform(-1, 1, 3)
    anw, notes, dydl = [], [], []
    S_1, S_2, dr, distance = None, None, None, None
    t = p.t if t is None else t
    g_all = []

    def get_U(obj, i, t):
        from dynamics import get_matrices
        U, S, A, R_orb = get_matrices(vrs=vrs, t=t, obj=obj, n=i)
        return U

    for obj1 in [c, f]:
        for obj2 in [f]:
            for i_1 in range(obj1.n):
                for i_2 in range(obj2.n) if obj1 == c else range(i_1):
                    # >>>>>>>>>>>> Расчёт положений и ориентаций <<<<<<<<<<<<
                    U_1 = float2rational(get_U(obj1, i_1, t), (1, 2), (1, 1))
                    U_2 = float2rational(get_U(obj2, i_2, t), (1, 2), (1, 1))
                    if produce:
                        dr = obj1.r_orf[i_1] - obj2.r_orf[i_2]
                        if isinstance(dr, np.ndarray):
                            p.record.loc[p.iter, f'{obj1.name}-{obj2.name} RealDistance {i_1} {i_2}'] = norm(dr)
                        A_1 = quart2dcm(obj1.q[i_1])
                        A_2 = quart2dcm(obj2.q[i_2])
                        S_1 = A_1 @ U_1.T
                        S_2 = A_2 @ U_2.T
                    else:
                        r1 = setvectype(estimated_params[i_1 * j + 0: i_1 * j + 3]) if obj1 == f else obj1.r_orf[i_1]
                        r2 = setvectype(estimated_params[i_2 * j + 0: i_2 * j + 3])
                        dr = r1 - r2
                        if vrs.NAVIGATION_ANGLES:
                            q1 = vec2quat(setvectype(estimated_params[i_1 * j + 3: i_1 * j + 6])) \
                                if obj1 == f else obj1.q[i_1]
                            q2 = vec2quat(setvectype(estimated_params[i_2 * j + 3: i_2 * j + 6]))
                            A_1 = quart2dcm(q1)
                            A_2 = quart2dcm(q2)
                            S_1 = A_1 @ U_1.T
                            S_2 = A_2 @ U_2.T
                        else:
                            A_1 = quart2dcm(obj1.q[i_1])
                            A_2 = quart2dcm(obj2.q[i_2])
                            S_1 = A_1 @ U_1.T
                            S_2 = A_2 @ U_2.T
                    d2 = dr[0]**2 + dr[1]**2 + dr[2]**2

                    # >>>>>>>>>>>> Расчёт G и сигнала <<<<<<<<<<<<
                    for direction in ["1->2"]:  # , "2->1"]:
                        take_len = len(get_gain(vrs=vrs, obj=obj2 if direction == "1->2" else obj1, r=randy))
                        send_len = len(get_gain(vrs=vrs, obj=obj2 if direction == "2->1" else obj1, r=randy))
                        G1 = [float2rational(g, (1, 2), (1, 1)) for g in get_gain(vrs=vrs, obj=obj1, r=S_1 @ dr)]
                        G2 = [float2rational(g, (1, 2), (1, 1)) for g in get_gain(vrs=vrs, obj=obj2, r=S_2 @ dr)]
                        # if not (isinstance(G1[0], int) or isinstance(G1[0], float)):
                        #     for i in range(len(G1)):
                        #         G1[i] = G1[i].subs([(Float(0.5), Rational(1, 2)), (Float(1.0), Rational(1, 1))])
                        # if not (isinstance(G2[0], int) or isinstance(G2[0], float)):
                        #     for i in range(len(G2)):
                        #         G2[i] = G2[i].subs([(Float(0.5), Rational(1, 2)), (Float(1.0), Rational(1, 1))])
                        g_vec = [g1 * g2 for g1 in G1 for g2 in G2]
                        g_all.extend(g_vec)

                        estimates = [gg / d2 for gg in g_vec] if not produce else \
                                    [(gg / d2) + np.random.normal(0, noise) for gg in g_vec]
                        est_dr = mean(estimates)
                        anw.extend(estimates)

                        # >>>>>>>>>>>> Расчёт производных по λ <<<<<<<<<<<<
                        '''if not produce and vrs.NAVIGATION_ANGLES and obj2.gain_mode != 'isotropic':
                            screw = get_antisymmetric_matrix
                            a = [np.array(i) for i in
                                 get_gain(vrs=vrs, obj=obj2, r=S_2 @ dr, return_dir=True)]
                            a = [a[i] for _ in G1 for i in range(len(G2))]
                            g1_vec = [g1 * 1 for g1 in G1 for _ in G2]
                            g2_vec = [1 * g2 for _ in G1 for g2 in G2]
                            e = dr / np.linalg.norm(dr)

                            tmp = [g1_vec[i] / d2 *
                                   (6 * (
                                       (screw(a[i]) @ S_2 @ e).T @ (-screw(a[i]) @ A_2 @ screw(U_2.T @ e))
                                   ) / g2_vec[i]**(2/3)) for i in range(len(g_vec))]
                            dydl.extend(tmp)'''

                        # >>>>>>>>>>>> Запись <<<<<<<<<<<<
                        o_fr, i_fr = (obj1, i_1) if direction == "1->2" else (obj2, i_2)
                        o_to, i_to = (obj1, i_1) if direction == "2->1" else (obj2, i_2)
                        if produce and isinstance(dr, np.ndarray):
                            p.record.loc[p.iter, f'{o_fr.name}-{o_to.name} EstimateDistance {i_fr} {i_to}'] = est_dr
                            p.record.loc[p.iter, f'{o_fr.name}-{o_to.name} ErrorEstimateDistance {i_fr} {i_to}'] = \
                                abs(est_dr - norm(dr))
                            p.record.loc[p.iter, f'{o_fr.name}-{o_to.name} ErrorEstimateDistance 1 {i_fr} {i_to}'] = \
                                abs(np.min(estimates) - norm(dr))
                            p.record.loc[p.iter, f'{o_fr.name}-{o_to.name} ErrorEstimateDistance 2 {i_fr} {i_to}'] = \
                                abs(np.max(estimates) - norm(dr))
                        notes.extend([f"{'c' if o_fr == c else 'f'}{'c' if o_to == c else 'f'} {i_fr} {i_to} {j} {i}"
                                      f" {send_len} {take_len}" for i in range(take_len) for j in range(send_len)])

    if produce:
        vrs.MEASURES_VECTOR = setvectype(anw)
        vrs.MEASURES_VECTOR_NOTES = notes
        if isinstance(dr, np.ndarray):
            p.record.loc[p.iter, f'G N'] = len(g_all)
            for i in range(len(g_all)):
                p.record.loc[p.iter, f'G {i}'] = g_all[i]
    else:
        return setvectype(anw), dydl, notes

def measure_magnetic_field(c: Apparatus, f: Apparatus, vrs: Variables, noise: float = 0.) -> None:
    """Функция обновляет для объектов CubeSat и FemtoSat параметры b_env"""
    import numpy as np
    for obj in [c, f]:
        for i in range(obj.n):
            obj.b_env[i] = np.zeros(3) + np.random.normal(0, noise, 3)

def measure_gps(f: Apparatus, noise: float) -> None:
    """Функция обновляет для объектов FemtoSat параметры _не_введено_"""
    pass
