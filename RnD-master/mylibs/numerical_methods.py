import random
from colorama import Fore, Style
from p_tqdm import p_map
import copy
from mylibs.tiny_functions import *


def e_combined(c: float, c_max: float, delta: float = 1e-2):
    if c/c_max <= 1 - delta:
        e = 1 - c/c_max + delta * c/c_max
    else:
        e = delta * np.exp(1 - c/c_max)
    return clip(e, 1e-4, 1e10)

def f_combined(e: float, dr: float = 5., mu: float = 1e-2):
    return dr**2 - mu*np.log(e)

def capturing_penalty(o, dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper, mu_ipm):
    """delta = 1e-1
    e_w = e_combined(c=abs(w), c_max=o.w_max, delta=delta)
    e_j = e_combined(c=abs(j), c_max=o.j_max, delta=delta)
    e_V = e_combined(c=abs(V), c_max=o.V_max, delta=delta)
    e_R = e_combined(c=abs(R), c_max=o.R_max, delta=delta)
    anw = dr * np.linalg.norm(dr)  # * (clip(1 - crhper, 0, 1) + clip(crhper, 0, 1) * tmp)
    anw += (-mu_ipm * (np.log(e_w) + np.log(e_V) + np.log(e_R) + np.log(e_j))) * dr / np.linalg.norm(dr)"""
    e_w = e_combined(c=abs(w), c_max=o.w_max)
    e_j = e_combined(c=abs(j), c_max=o.j_max)
    e_V = e_combined(c=abs(V), c_max=o.V_max)
    e_R = e_combined(c=abs(R), c_max=o.R_max)
    # print(f"w:{e_w}({np.log(e_w)}), j:{e_j}({np.log(e_j)}), V:{e_V}({np.log(e_V)}), R:{e_R}({np.log(e_R)})")
    anw = np.linalg.norm(dr) ** 2 - mu_ipm * (np.log(e_w) + np.log(e_V) + np.log(e_R) + np.log(e_j))
    return anw

def detour_penalty(o, dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper, mu_ipm):
    '''# Constraint's fulfillment
    if False:  # (o.e_max - e > 0) and (o.V_max - V > 0) and (o.R_max - R > 0) and (o.j_max - j > 0):
        anw = dr + dr_average - o.mu_ipm * (np.log(o.e_max - e) + np.log(o.V_max - V) +
                               np.log(o.R_max - R) + np.log(o.j_max - j)) + \
              np.array([1e3, 1e3, 1e3]) * n_crashes'''

    anw = np.linalg.norm(dr) + dr_average * 0.1 + 1 * n_crashes + 1e2 * (not visible)
    params = [[o.e_max, e], [o.j_max, j], [o.V_max, V], [o.R_max, R]]
    for i in range(2):
        if params[i][0] - params[i][1] > 0:  # Constraint's fulfillment
            anw -= o.mu_ipm * np.log(params[i][0] - params[i][1])
        else:
            anw += 1e2
    # print(f"{n_crashes} {1e2 * visible} |  {np.linalg.norm(dr)}:{dr_average}  | {np.linalg.norm(anw)}")
    return anw

def f_scipy(u, *args):
    from mylibs.calculation_functions import calculation_motion
    o, T_max, id_app, interaction, check_visible, mu_ipm = args
    f_min, e_max, V_max, R_max, w_max, j_max, g = calculation_motion(o=o, u=u, T_max=T_max, id_app=id_app,
                                                                     interaction=interaction, to_scipy=True)

def f_dr(u, *args):
    from mylibs.calculation_functions import calculation_motion
    o, T_max, id_app, interaction, check_visible, mu_ipm = args
    dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper = calculation_motion(o=o, u=u, T_max=T_max, id_app=id_app,
                                                                                   interaction=interaction,
                                                                                   check_visible=False)
    return np.linalg.norm(dr)**2

def f_to_capturing(u, *args) -> tuple:
    from mylibs.calculation_functions import calculation_motion
    if len(args) == 1:
        args = args[0]
    o, T_max, id_app, interaction, check_visible, mu_ipm = args
    u = o.cases['repulse_vel_control'](u)
    dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper = calculation_motion(o=o, u=u, T_max=T_max, id_app=id_app,
                                                                                   interaction=interaction,
                                                                                   check_visible=False)
    return capturing_penalty(o, dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper, mu_ipm), np.linalg.norm(dr)

def f_to_detour(u, *args):
    from mylibs.calculation_functions import calculation_motion
    if len(args) == 1:
        args = args[0]
    o, T_max, id_app, interaction, check_visible, mu_ipm = args
    u = o.cases['repulse_vel_control'](u)
    dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper = calculation_motion(o, u, T_max, id_app,
                                                                                   interaction=interaction,
                                                                                   check_visible=check_visible)
    anw = detour_penalty(o, dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper, mu_ipm)
    return anw

def f_controlled_const(v, *args) -> tuple:
    from mylibs.calculation_functions import calculation_motion
    if len(args) == 1:
        args = args[0]
    o, T_max, id_app, interaction, check_visible, mu_ipm = args
    u = o.cases['repulse_vel_control'](v[0:3])
    if o.if_T_in_shooting:
        tmp = len(v) - 1
    else:
        tmp = len(v) 
    dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper = calculation_motion(o, u, T_max, id_app,
                                                                                   interaction=interaction,
                                                                                   check_visible=check_visible,
                                                                                   control=v[3:tmp])
    return capturing_penalty(o, dr, dr_average, e, V, R, w, j, n_crashes, visible, crhper, mu_ipm)

def my_calc_shooting(o, id_app, r_1, interaction: bool = True, u: any = None, func: any = f_dr):
    """u = [T, u_x, u_y, u_z]
    ШАМАНСТВО: что это такое? ЧТО ТЫ ДЕЛАЕШЬ?"""
    shooting_amount = o.shooting_amount_repulsion if interaction else o.shooting_amount_impulse
    i_iteration = 0

    def local_correction(t, w, v_x, v_y, v_z, x0, y0, z0, x1, y1, z1):
        from numpy import sin, cos
        return np.array([2*(-(3*t*w - 4*sin(t*w))*(2*v_z*cos(t*w) - 2*v_z - w*(3*t*(v_x + 2*w*z0) - x0 + x1) +
                                                   (4*v_x + 6*w*z0)*sin(t*w)) -
                            2*(cos(t*w) - 1)*(2*v_x + v_z*sin(t*w) + w*(4*z0 - z1) +
                                              (-2*v_x - 3*w*z0)*cos(t*w)))/w**2,
                         2*(-(3*t*w - 4*sin(t*w))*(2*v_z*cos(t*w) - 2*v_z - w*(3*t*(v_x + 2*w*z0) - x0 + x1) +
                                                   (4*v_x + 6*w*z0)*sin(t*w)) - 2*(cos(t*w) - 1) *
                            (2*v_x + v_z*sin(t*w) + w*(4*z0 - z1) + (-2*v_x - 3*w*z0)*cos(t*w)))/w**2,
                         2*(v_y*sin(t*w) + w*(y0*cos(t*w) - y1))*sin(t*w)/w**2,
                         2*(2*(cos(t*w) - 1)*(2*v_z*cos(t*w) - 2*v_z - w*(3*t*(v_x + 2*w*z0) - x0 + x1) +
                                              (4*v_x + 6*w*z0)*sin(t*w)) + (2*v_x + v_z*sin(t*w) + w*(4*z0 - z1) +
                                                                            (-2*v_x - 3*w*z0)*cos(t*w))*sin(t*w))/w**2])

    while i_iteration < shooting_amount:
        i_iteration += 1
        f = func(u[1:4], o, u[0], id_app, interaction, False, o.mu_ipm)
        print(f"Итерация {i_iteration}|dr={round(np.sqrt(f),5)}, T={u[0]}, u={u[1:4]}")
        if f > o.d_to_grab**2:
            v = o.b_o(u[1:4])
            r0 = o.b_o(o.a.target_p[id_app])
            r1 = o.b_o(r_1)
            u -= local_correction(u[0], o.w_hkw, v[0], v[1], v[2], r0[0], r0[1], r0[2], r1[0], r1[1], r1[2])
        pass

def calc_shooting_sample(u, o, T_max, id_app, interaction, mu_ipm, func):
    """Функция, подаваемая на случайное ядро процессора при распараллеливании. Интегрирует уравнения движения"""
    T_max = u[len(u) - 1] if o.if_T_in_shooting else T_max
    return func(u, o, T_max, id_app, interaction, o.method in ['linear-propulsion', 'const-propulsion'], mu_ipm)

def calc_shooting(o, id_app, r_1, interaction: bool = True, u0: any = None, n: int = 3, func: any = f_to_capturing,
                  T_max=None) -> tuple:
    """Функция выполняет пристрелочный поиск подходящей скорости отталкивания/импульса аппарата
    :param o: AllObjects класс
    :param id_app: id-номер аппарата
    :param r_1: радиус-вектор ССК положения цели
    :param interaction: происходит ли при импульсе отталкивание аппарата от конструкции
    :param u0: начальное приближение
    :param n: длина рабочего вектора
    :param func: функция, выдающая вектор длины 3, минимизируемая оптимальным вход-вектором
    :param T_max: время полёта
    :return u: скорость отталкивания/импульса аппарата
    :return cond: подходящая ли скорость u"""
    shooting_amount = o.shooting_amount_repulsion if interaction else o.shooting_amount_impulse  # Число итераций
    tmp = r_1 - o.o_b(o.a.r[id_app]) if interaction else o.b_o(r_1) - np.array(o.a.r[id_app])  # Начальная невязка
    T_max = o.T_max if T_max is None else T_max
    u = o.u_min * tmp / np.linalg.norm(tmp) if u0 is None else u0  # Начальное приближение "тупо на цель" если u0=None
    if n > 3:
        u = np.zeros(n)
        u[0:3] = o.u_min * tmp / np.linalg.norm(tmp) if u0 is None or np.linalg.norm(u0) < 1e-5 else u0[0:3]
    u_anw = u.copy()  # Будущий return
    mu_ipm = o.mu_ipm  # Коэффициент μ уменьшается с каждой итерацией
    i_iteration = 0  # На случай условия добавочных итераций (например, при невыполнении условий)
    n_full = n

    # Запись параметров
    tol_anw = 1e3
    tol_list = []

    # Вариация начальной скорости
    dd = 1e-4
    du = dd * o.u_max
    variation = [du] * 3 + [dd * o.a_pid_max] * (n - 3)
    if o.if_T_in_shooting:
        u = np.append(u, T_max)
        n_full += 1
        variation += [1.]
    u_i = np.diag(variation)

    # Метод пристрелки
    while i_iteration < shooting_amount:
        # Изменение параметров
        mu_ipm /= 1.2
        i_iteration += 1

        # Расчёт
        # [центр, [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
        anw = p_map(calc_shooting_sample,
                    [u] +
                    [u + sum([u_i[:, j]*(int(np.binary_repr(i, width=3)[j])*2-1) for j in range(3)]) for i in range(8)]
                    + [u + u_i[:, i] for i in range(3)] + [u - u_i[:, i] for i in range(3)],
                    [o for _ in range(1 + 8 + 6)],
                    [T_max for _ in range(1 + 8 + 6)],
                    [id_app for _ in range(1 + 8 + 6)],
                    [interaction for _ in range(1 + 8 + 6)],
                    [mu_ipm for _ in range(1 + 8 + 6)],
                    [func for _ in range(1 + 8 + 6)])
        tol_list += [anw[0][1]]
        f0 = anw[0][0]
        f_cube = [anw[1+i][0] for i in range(8)]
        f_cross = [anw[1+8+i][0] for i in range(6)]
        # print(f"f0={f0}; f_cube={f_cube}; f_cross={f_cross}")
        o.my_print(f"Точность {round(tol_list[-1],5)}, целевая функция {round(f0, 5)}, T_max={T_max}", mode='c')
        if np.linalg.norm(f0) < tol_anw:
            tol_anw = np.linalg.norm(f0)
            u_anw = u.copy()
   
        if o.d_to_grab is not None and tol_list[-1] < o.d_to_grab*0.98:  # Выполнение условий, критерий останова
            u_anw = u.copy()
            break
        else:
            # [центр, [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
            # Расчёт ∇Ф: 4 стороны квадрата - 4 стороны квадрата
            # x: 5+6+7+8 - 1-2-3-4
            # y: 3+4 + 7+8 - 1-2 - 5-6
            # z: 2+4+6+8 - 1-3-5-7
            gradient = np.array([
                (f_cross[i] - f_cross[3+i]) / (2 * du)
                for i in range(3)])
            """gradient = np.array([
                np.array(f_cube).dot([int(np.binary_repr(k, width=3)[i])*2-1 for k in range(8)]) / (4 * 2 * du)
                for i in range(3)])"""
            Gramian = np.array([[
                (f_cross[i] + f_cross[3+i] - 2*f0) / (du**2)
                if i == j else
                np.array(f_cube).dot([int(np.binary_repr(k, width=3)[i] == np.binary_repr(k, width=3)[j])*2-1
                                      for k in range(8)]) / (2 * 2 * 2 * du**2)
                for j in range(n)] for i in range(3)])
            # print(f"gradient={gradient}\n\nGramian={Gramian}\n")

            if n == 3 and abs(np.linalg.det(Gramian)) > 1e-10:
                Gramian_1 = np.linalg.inv(Gramian)
            else:
                print(f"Используется псевдообратная Матрица Гессе! ΔG={np.linalg.det(Gramian)}")
                # Gramian_1 = np.linalg.pinv(Gramian)
                break  # to_delete
            correction = Gramian_1 @ gradient
            # o.my_print(f"ΔG={np.linalg.det(Gramian)}, ∇Ф={np.linalg.norm(gradient)} --> {np.linalg.norm(Gramian_1 @ gradient)}", mode='c')

            if np.linalg.norm(correction) > 1e-12:
                correction = correction / np.linalg.norm(correction) * clip(np.linalg.norm(correction), 0, o.u_max / 4)
            else:
                print(f"Пристрелка закончена, градиент/грамиан нулевой")
                break
            u -= correction

            # Ограничение по модулю скорости
            if np.linalg.norm(u[0:3]) > o.u_max:
                o.my_print(f'attention: shooting speed reduction: {np.linalg.norm(u[0:3])/o.u_max*100} %')
                u[0:3] = u[0:3] / np.linalg.norm(u[0:3]) * o.u_max * 0.9
            if o.method != 'linear-angle':
                if n > 3 and np.linalg.norm(u[3:6]) > o.a_pid_max:
                    o.my_print(f'attention: acceleration reduction: {np.linalg.norm(u[3:6])/o.a_pid_max*100} %')
                    u[3:6] = u[3:6] / np.linalg.norm(u[3:6]) * o.a_pid_max * 0.9
                if n > 6 and np.linalg.norm(u[6:9]) > o.a_pid_max:
                    o.my_print(f'attention: acceleration reduction: {np.linalg.norm(u[6:9])/o.a_pid_max*100} %')
                    u[6:9] = u[6:9] / np.linalg.norm(u[6:9]) * o.a_pid_max * 0.9

            '''if np.linalg.norm(dr) > tol * 1e2 and i_iteration == shooting_amount:
                mu_ipm *= 10
                i_iteration -= 1
                T_max *= 1.1 if T_max < o.T_max_hard_limit else 1'''

        T_max = u[len(u) - 1] if o.if_T_in_shooting else T_max
    file_local = open('storage/iteration_docking.txt', 'a')
    if True:  # tol < o.d_to_grab*0.999:
        for i in range(len(tol_list)):
            file_local.write(f"{i} {tol_list[i]}\n")
    file_local.close()
    return u_anw, tol_list[-1] < o.d_to_grab*0.98

def diff_evolve_sample(j: int, func: any, v, target_p, comp_index: list,
                       chance: float = 0.5, f: float = 1., *args):
    args = args[0]
    mutant = v[comp_index[0]].copy() + f * (v[comp_index[1]] - v[comp_index[2]])
    for i in range(len(mutant)):
        if random.uniform(0, 1) < chance:
            mutant[i] = v[j][i].copy()
    target_p = func(v[j], args) if target_p is None else target_p
    target = func(mutant, args)
    v[j] = mutant.copy() if target < target_p else v[j]
    target_p = target if target < target_p else target_p
    return np.append(v[j], target_p)

def diff_evolve(func: any, search_domain: list, vector_3d: bool = False, *args, **kwargs):
    """Функция дифференциальной эволюции.
    :param func: целевая функция
    :param search_domain: 2-мерный список разброса вектора аргументов: [[v[0]_min, v[0]_max], [v[1]_min, v[1]_max],...]
    :param vector_3d: bla bla bla
    :return: v_best: len_vec-мерный список"""
    chance = 0.5 if 'chance' not in kwargs.keys() else kwargs['chance']
    f = 1. if 'f' not in kwargs.keys() else kwargs['f']
    n_vec = 10 if 'n_vec' not in kwargs.keys() else kwargs['n_vec']
    len_vec = 3 if 'len_vec' not in kwargs.keys() else kwargs['len_vec']
    n_times = 5 if 'n_times' not in kwargs.keys() else kwargs['n_times']
    multiprocessing = True if 'multiprocessing' not in kwargs.keys() else kwargs['multiprocessing']
    print_process = False if 'print_process' not in kwargs.keys() else kwargs['print_process']
    lst_errors = []
    # попробовать tuple(search_domain[i])
    if vector_3d:
        if len_vec == 3:
            v = np.array([polar2dec(np.exp(random.uniform(np.log(search_domain[0]), np.log(search_domain[1]))),
                                    random.uniform(0, 2 * np.pi),
                                    random.uniform(- np.pi / 2, np.pi / 2)) for _ in range(n_vec)])
        else:
            v = np.array([np.append(polar2dec(np.exp(random.uniform(np.log(search_domain[0]), np.log(search_domain[1]))),
                                              random.uniform(0, 2 * np.pi), random.uniform(- np.pi / 2, np.pi / 2)),
                                    [random.uniform(search_domain[2], search_domain[3])
                                     for _ in range(len_vec - 3)]) for _ in range(n_vec)])
    else:
        v = np.array([np.array([random.uniform(search_domain[i][0], search_domain[i][1]) for i in range(len_vec)])
                      for _ in range(n_vec)])
    v_record = [copy.deepcopy(v)]
    target_prev = [None for _ in range(n_vec)]
    v_best = None
    for i in range(n_times):
        if print_process:
            print(Fore.CYAN + f"Шаг {i + 1}/{n_times} дифференциальной эволюции" + Style.RESET_ALL)
        comp_index = [[] for _ in range(n_vec)]
        for j in range(n_vec):
            complement = list(range(n_vec))
            complement.remove(j)
            for _ in range(3):
                comp_index[j].append(random.choice(complement))
                complement.remove(comp_index[j][len(comp_index[j]) - 1])
        anw = p_map(diff_evolve_sample,
                    [j for j in range(n_vec)],
                    [func for _ in range(n_vec)],
                    [v for _ in range(n_vec)],
                    [target_prev[j] for j in range(n_vec)],
                    [comp_index[j] for j in range(n_vec)],
                    [chance for _ in range(n_vec)],
                    [f for _ in range(n_vec)],
                    [args for _ in range(n_vec)]) if multiprocessing else \
            [diff_evolve_sample(j, func, v,
                                target_prev[j],
                                comp_index[j],
                                chance, f, args) for j in range(n_vec)]
        v = np.array([np.array(anw[j][0:len_vec]) for j in range(n_vec)])
        v_record += [copy.deepcopy(v)]
        target_prev = [anw[j][len_vec] for j in range(n_vec)]
        lst_errors.append(np.min(target_prev))
        v_best = v[np.argmin(target_prev)]
    print(Fore.MAGENTA + f"Ошибка: {lst_errors}" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"Ответ: {v_best}" + Style.RESET_ALL)
    # plt.plot(range(len(lst_errors)), lst_errors, c='indigo')
    # plt.show()
    return v_best, v_record
