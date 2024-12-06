from mylibs.im_sample import *
from mylibs.numerical_methods import *
from mylibs.control_function import control_condition, force_from_beam, simple_control
import scipy


def call_crash_internal_func(r, r1, r2, diam, return_force=False, k_av=None, level: int = 5, return_dist=False):
    """ Дополнительная функция для функции call_crash; \n
    Проверяет наличие точки r в цилиндре с концами r1,r2, диаметром diam; \n
    Возвращает {True,False} соответственно при отсутствии и наличии. """
    r1 = np.array(r1.copy())
    r2 = np.array(r2.copy())
    x1, y1, z1 = r1
    x2, y2, z2 = r2
    n = np.array([x2 - x1, y2 - y1, z2 - z1])  # Вектор вдоль стержня
    tau = my_cross(np.array([0, 0, 1]), n)
    if np.linalg.norm(tau) < 1e-6:
        tau = my_cross(np.array([1, 0, 0]), n)
        if np.linalg.norm(tau) < 1e-6:
            tau = my_cross(np.array([0, 1, 0]), n)
    b = my_cross(tau, n)
    a = r - (r1 + r2) / 2
    f0 = np.dot(a, n) / (np.linalg.norm(n)**2 / 2)
    f1 = np.dot(a, tau) / (np.linalg.norm(tau) * diam)
    f2 = np.dot(a, b) / (np.linalg.norm(b) * diam)
    '''if return_dist:
        reserve = 0.
        if (f0 > -1) and (f0 < 1):
            return np.sqrt(f1**2 + f2**2) - 1 - reserve
        elif f1**2 + f2**2 < 1:
            return (abs(f0) - 1) - reserve
        else:
            return (np.sqrt(f1 ** 2 + f2 ** 2) - 1) * 1 +\
                (abs(f0) - 1) - reserve'''
    if return_dist:
        reserve = 0.
        if (f0 > -1) and (f0 < 1):
            return (np.sqrt(f1**2 + f2**2) - 1) * diam - reserve 
        elif f1**2 + f2**2 < 1:
            return (abs(f0) - 1) * (np.linalg.norm(n) / 2) - reserve  # * np.sign(f0)
        else:
            return (np.sqrt(f1 ** 2 + f2 ** 2) - 1) * diam +\
                (abs(f0) - 1) * (np.linalg.norm(n) / 2) - reserve
    if return_force:
        return force_from_beam(a, diam, n, tau, b, f0, f1, f2, k_av, level)
    if not ((f0 > -1) and (f0 < 1) and (f1**2 + f2**2 < 1)):
        return False
    return True


def call_crash(o, r_sat, R, S, taken_beams=np.array([]), iFunc=False, brf=False):
    """ Функция проверяет наличие тела внутри балки (назовём это соударением);      \n
    Input:                                                                          \n
    -> o - AllObjects класс;                                                        \n
    -> r_sat - радиус-вектор центра аппарата; (отдельно для эпизодов)               \n
    -> R - вектор до центра масс конструкции; (отдельно для эпизодов)               \n
    -> S - матрица поворота ОСК -> ССК;       (отдельно для эпизодов)               \n
    -> taken_beams - вектор id стержней, которых учитывать не надо                  \n
    Output:                                                                         \n
    -> True в случае соударения, False иначе. """
    if not iFunc:
        if o.d_crash is None:
            return False
        r = r_sat if brf else o.o_b(r_sat, S=S, R=R)
        for i in range(o.s.n_beams):
            last_id = 8 + i * 4 + (o.s.floor + 1) * 4
            if o.choice == '4' and i < last_id or o.choice != '4':  # прикол на большой круг  контсрукции 4
                if not(np.any(taken_beams == i)) and np.sum(o.s.flag[i]) > 0:
                    r1 = o.s.r1[i]
                    r2 = o.s.r2[i]
                    if call_crash_internal_func(r, r1, r2, o.d_crash):
                        return True
        if o.choice == '4':  # собсна большой круг
            ext = call_crash_internal_func(r, np.array([0., o.s.big_length / 2, 0.]), 
                                           np.array([0., - o.s.big_length / 2, 0.]), o.s.r_big_circle + o.s.big_length)
            int = call_crash_internal_func(r, np.array([0., o.s.big_length / 2, 0.]), 
                                           np.array([0., - o.s.big_length / 2, 0.]), o.s.r_big_circle)
            if not int and ext:
                return True

        r1 = o.c.r1[0]
        r2 = o.c.r2[0]
        if call_crash_internal_func(r, r1, r2, o.c.diam[0] * 1.01):
            return True
        if call_crash_internal_func(r, np.zeros(3), np.array([-o.s.x_start - o.s.container_length, 0., 0.]),
                                    o.c.r_around + o.d_crash):
            return True
        return False
    else:
        if o.d_crash is None:
            return 1.
        r = o.o_b(r_sat, S=S, R=R)
        g = []
        for i in range(o.s.n_beams):
            if not(np.any(taken_beams == i)):
                if np.sum(o.s.flag[i]) > 0:
                    r1 = o.s.r1[i]
                    r2 = o.s.r2[i]
                    g += [call_crash_internal_func(r, r1, r2, o.d_crash, return_dist=True)]
        r1 = o.c.r1[0]
        r2 = o.c.r2[0]
        g += [call_crash_internal_func(r, r1, r2, o.c.diam[0] * 1.01, return_dist=True)]
        g += [call_crash_internal_func(r, np.zeros(3), np.array([-o.s.x_start - o.s.container_length, 0., 0.]),
                                       o.c.r_around + o.d_crash, return_dist=True)]
        return np.min(g)


def call_inertia(o, id_s, app_y=None, app_n=None) -> tuple:
    """Функция считает тензор инерции в собственной ск конструкции и центр масс;    \n
    Input:                                                                          \n
    -> o - AllObjects класс;                                                        \n
    -> id_s - вектор id стержней, которых учитывать не надо                         \n
    -> app_y - номера аппаратов, которых стоит учесть при расчёте инерции;          \n
    -> app_n - номера аппаратов, которых не стоит учитывать при расчёте инерции;    \n
    Output:                                                                         \n
    -> Тензор инерции, вектор центра масс. """
    J = np.zeros((3, 3))
    r = np.zeros(3)
    m = 0.

    # Ферма
    for n in range(o.s.n):
        flag = not np.any(id_s == n)
        if o.main_numerical_simulation and n == 0:
            print(f"++++++++++++++++ {flag} id={n}")
        """for a in range(o.a.n):
            if o.a.flag_beam[a] is not None and int(o.a.flag_beam[a]) == n:
                flag = False"""
        if flag:
            if np.sum(o.s.flag[n]) > 0:
                r_1 = np.array(o.s.r1[n])
                r_2 = np.array(o.s.r2[n])
            else:
                r_1 = o.s.r_st[n]
                r_2 = o.s.r_st[n] - np.array([o.s.length[n], 0, 0])
            r += o.s.mass[n] * (0.5*r_1 + 0.5*r_2)
            m += o.s.mass[n]
            J += o.s.mass[n] * local_tensor_of_rod(r_1, r_2)

    # Грузовой отсек + каркас
    for n in range(o.c.n):
        r_1 = np.array(o.c.r1[n])
        r_2 = np.array(o.c.r2[n])
        r += o.c.mass[n] * (r_1 + r_2) / 2
        m += o.c.mass[n]
        J += o.c.mass[n] * local_tensor_of_rod(r_1, r_2) if n > 0 else o.c.mass[n] * np.array(
            [[(o.c.diam[n]/2) ** 2 / 2, 0, 0],
             [0, (o.c.diam[n]/2) ** 2 / 4 + np.linalg.norm(r_1 - r_2) ** 2 / 12, 0],
             [0, 0, (o.c.diam[n]/2) ** 2 / 4 + np.linalg.norm(r_1 - r_2) ** 2 / 12]])

    # Сборочный КА
    for a in range(o.a.n):
        if (not o.a.flag_fly[a] and app_n != a) or app_y == a:
            tmp = o.a.mass[a]  # if o.a.flag_beam[a] is None else o.a.mass[a] + o.s.mass[int(o.a.flag_beam[a])]
            r += tmp * o.a.target_p[a]
            m += tmp
            J += local_tensor_of_point(tmp, o.a.target_p[a])
            # print(f"r_a = {o.a.target_p[a]}")
        # print(f"~~~~~~~~~~ a: {(not o.a.flag_fly[a] and app_n != a) or app_y == a} (m_a = {o.a.mass[a]}) --> m_ub = {m}")

    # Расчёт центра масс + теорема Гюйгенца-Штейнера
    r /= m
    J -= local_tensor_of_point(m, r)
    return J, r


def call_middle_point(X_cont, r_1):
    """ Функция возвращает id удобной ручки грузового отсека как промежуточная цель;    \n
    Input:                                                                              \n
    -> информационная матрица X_cont, содержашая столбцы:                               \n
    id | mass | length | diam | r1 | r2 | flag_grab                                     \n
    -> r_1 - радиус-вектор целевой точки на конструкции СтСК                            \n
    Output:                                                                             \n
    -> id-номер удобной ручки крепления с конструкцией """
    X_middles = []
    for i in range(len(X_cont.id)):
        if X_cont.flag_grab[i]:
            X_middles.append(i)
    difference = [X_cont.r1[X_middles[i]] / 2 + X_cont.r2[X_middles[i]] / 2 for i in range(len(X_middles))]
    N_middles = len(difference)
    difs = np.zeros(N_middles)
    for i in range(N_middles):
        difs[i] = np.linalg.norm(difference[i] - r_1)
    i_needed = np.argmin(difs)

    return np.array(X_middles)[i_needed]


def calculation_motion(o, u, T_max, id_app, interaction=True, line_return=False, check_visible=False, control=None,
                       to_scipy=False):
    """Функция расчитывает угловое и поступательное движение одного аппарата и конструкции; \n
    Других аппаратов и их решения она не учитывает;                                         \n
    Input:                                                                                  \n
    -> o - AllObjects класс;                                                                \n
    -> u - начальная скорость аппарата (ССК при interaction=True, ОСК иначе)                \n
    -> T_max - время эпизода                                                                \n
    -> id_app - номер аппарата                                                              \n
    -> interaction - происходит ли при импульс99е отталкивание аппарата от конструкции        \n
    -> line_return - возвращать ли линию полёта аппарата ССК (для красивых картинок)        \n
    Output:                                                                                     \n
    -> F_min - минимальная невязка-вектор от нужной точки до аппарата                           \n
    -> F_min - средняя невязка-вектор от нужной точки до аппарата                               \n
    -> w_modeling/w_after - максимальный модуль угловой скорости станции по ходу/после эпизода  \n
    -> V_max - максимальный модуль поступательной скорости станции по ходу/после эпизода        \n
    -> R_max - максимальный модуль положения ОСК станции по ходу эпизода                        \n
    -> j_max - максимальный угол отклонения от целевого положения станции по ходу эпизода       \n
    -> N_crashes - количество соударений (не должно превосходить 1)                             \n
    -> line - линия аппарата ССК (при line_return=True)"""
    # time_start = datetime.now()
    o_lcl = o.copy()
    # print(f"копирование: {datetime.now() - time_start}")

    e_max, V_max, R_max, w_max, j_max, f_mean = np.zeros(6)
    o_lcl.flag_vision[id_app] = False
    if control is None or len(control) == 0:
        control = None
    else:
        o_lcl.a.flag_hkw[id_app] = False
    n_crashes = 0
    crah_percantage = 0.
    line = []
    g = []

    # time_start = datetime.now()
    if interaction:  
        o_lcl.repulsion_change_params(id_app=id_app, u0=u)  # o.cases['repulse_vel_control'](u))
    # print(f"начало: {datetime.now() - time_start}")

    f_min = o_lcl.S @ o_lcl.get_discrepancy(id_app=id_app, vector=True) if interaction else \
        o_lcl.get_discrepancy(id_app=id_app, vector=True)

    # ШАМАНСТВО
    check_visible = False

    # Iterations
    # time_start = datetime.now()
    for i in range(int(T_max // o_lcl.dt)):
        # Writing parameters
        f = o_lcl.S @ o_lcl.get_discrepancy(id_app=id_app, vector=True) if interaction else \
            o_lcl.get_discrepancy(id_app=id_app, vector=True)
        if check_visible:
            _, tmp = control_condition(o_lcl, id_app, return_percentage=True)
            # crah_percantage = max(crah_percantage, tmp)
        # ВЫ ЗАБЫЛИ ЛИЦА СВОИХ ОТЦОВ И ИХ ОТЦОВ
        # ВЫ ЗАБЫЛИ САМИХ СЕБЯ
        if True:  # np.linalg.norm(f_min) > np.linalg.norm(f):
            f_min = f.copy()
            # crah_percantage = tmp if check_visible else crah_percantage
        e_max = max(e_max, o_lcl.get_e_deviation())
        V_max = max(V_max, np.linalg.norm(o_lcl.v_ub))
        R_max = max(R_max, np.linalg.norm(o_lcl.r_ub))
        w_max = max(w_max, np.linalg.norm(o_lcl.w))
        j_max = max(j_max, 180/np.pi*np.arccos((np.trace(o_lcl.S)-1)/2))
        f_mean += np.linalg.norm(f)
        line = np.append(line, o_lcl.o_b(o_lcl.a.r[id_app]))

        # Stop Criteria
        if to_scipy and (o_lcl.t - o.t) > o.freetime:
            # g += [call_crash(o, o_lcl.a.r[id_app], o_lcl.R, o_lcl.S, o_lcl.taken_beams, iFunc=True)]
            g += [o_lcl.o_b(o_lcl.a.r[id_app])]
        else:
            if o_lcl.d_to_grab is not None:
                if np.linalg.norm(f) <= o_lcl.d_to_grab * 0.95:
                    break
            if o_lcl.d_crash is not None:
                if (o_lcl.t - o.t) > o.freetime:
                    if call_crash(o, o_lcl.a.r[id_app], o_lcl.r_ub, o_lcl.S, o_lcl.taken_beams):
                        n_crashes += 1
                        break
            if not o_lcl.control and \
                    np.linalg.norm(o_lcl.a.r[id_app] - o_lcl.b_o(o_lcl.a.target_p[id_app])) > 15 * o_lcl.a.r_0[id_app]:
                break
            '''if i % 5 == 0:
                o_lcl.file_save(f'график {id_app} {np.linalg.norm(f)} {np.linalg.norm(o_lcl.w)} '
                                f'{np.linalg.norm(180 / np.pi * np.arccos(clip((np.trace(o_lcl.S) - 1) / 2, -1, 1)))} '
                                f'{np.linalg.norm(o_lcl.v_ub)} {np.linalg.norm(o_lcl.r_ub)} '
                                f'{np.linalg.norm(o_lcl.a_self[id_app])}')'''

        # Iterating
        if control is not None:
            o_lcl.a_self[id_app] = simple_control(o, control, (o_lcl.t - o.t) / T_max)
        o_lcl.time_step()
    # print(f"потрачено на итерации: {datetime.now() - time_start}")
    anw_dr = o_lcl.o_b(o_lcl.a.r[id_app])

    if check_visible:
        _, crah_percantage = control_condition(o_lcl, id_app, return_percentage=True)
        # crah_percantage = max(crah_percantage, tmp)
        # print(f"----{tmp_a} {o_lcl.flag_vision[id_app]}")

    if (o_lcl.iter - o.iter) > 0:
        f_mean /= (o_lcl.iter - o.iter)

    # ШАМАНСТВО
    if True:  # interaction is False:   # Пристыковка
        o_lcl.capturing_change_params(id_app)
        V_max = np.linalg.norm(o_lcl.v_ub)
        R_max = np.linalg.norm(o_lcl.r_ub)
        w_max = np.linalg.norm(o_lcl.w)
        e_max = o_lcl.get_e_deviation()

        for i in range(int((o_lcl.iter - o.iter) // 20)):
            o_lcl.time_step()

            e_max = max(e_max, o_lcl.get_e_deviation())
            V_max = max(V_max, np.linalg.norm(o_lcl.v_ub))
            R_max = max(R_max, np.linalg.norm(o_lcl.r_ub))
            j_max = max(j_max, 180/np.pi*np.arccos((np.trace(o_lcl.S)-1)/2))
            w_max = max(w_max, np.linalg.norm(o_lcl.w))

    if to_scipy:
        return anw_dr, e_max, V_max, R_max, w_max, j_max, g
    anw = (f_min, f_mean, e_max, V_max, R_max, w_max, j_max, n_crashes, o_lcl.flag_vision[id_app], crah_percantage,)
    if line_return:
        anw += (line,)
    return anw


def find_repulsion_velocity(o, id_app: int, target=None, interaction: bool = True, method: str = 'trust-constr',
                            u: any = None, T_max: float = None):
    # if interaction and not o.control or not interaction:
    func = f_dr
    tol = o.d_to_grab
    o.my_print(f"Попадание: tol={tol}", test=True)
    '''else:
        func = f_dr
        tol = o.a.r_0[0]
        o.my_print(f"Огибание: tol={tol}, cont={o.s.container_length}", test=True)'''

    if interaction:
        target = o.a.target[id_app] if target is None else target
        tmp = target - np.array(o.o_b(o.a.r[id_app]))
    else:
        target = o.b_o(o.a.target[id_app]) if target is None else target
        tmp = o.b_o(target) - np.array(o.a.r[id_app])
    mtd = method  # 'SLSQP' 'TNC' 'trust-constr'
    opt = {'verbose': 3}
    u = o.u_min * tmp / np.linalg.norm(tmp) if u is None else u
    T_max = o.T_max if T_max is None else T_max

    res = scipy.optimize.minimize(func, u, args=(o, T_max, id_app, interaction, False, o.mu_ipm),
                                  tol=tol, method=mtd, options=opt,
                                  bounds=((0, o.u_max), (0, o.u_max), (0, o.u_max)),
                                  constraints={'type': 'ineq',
                                               'fun': lambda x: (np.linalg.norm(x) - o.u_min) / (o.u_max - o.u_min)})
    u = o.cases['repulse_vel_control'](res.x)
    return u

def find_repulsion_velocity_new(o, id_app: int, target=None, interaction: bool = True,
                                u: any = None, T_max: float = None, ifunc=True):
    tol = o.d_to_grab**2
    T_max = o.T_max if T_max is None else T_max
    target = o.a.target[id_app] if target is None else target
    opt = {'verbose': 3, 'gtol': 1e-20, 'xtol': 1e-20}
    if ifunc:
        s = np.append(u, target)
        print(f"s0={s}")
        bounds = ((-o.u_max/2, o.u_max/2), (-o.u_max/2, o.u_max/2), (-o.u_max/2, o.u_max/2),
                  (-1e2, 1e2), (-1e2, 1e2), (-1e2, 1e2))
    else:
        s = u.copy()
        bounds = ((-o.u_max/2, o.u_max/2), (-o.u_max/2, o.u_max/2), (-o.u_max/2, o.u_max/2))
    o.my_print(f"Попадание: tol={tol}, T={T_max} s={s}, r_1={target}", test=True)

    def local_func(s1):
        if ifunc:
            return np.linalg.norm(target - s1[3:6])**2
        else:
            dr, e_max, V_max, R_max, w_max, j_max, g = calculation_motion(o=o, u=s1, T_max=T_max, id_app=id_app,
                                                                          interaction=interaction, to_scipy=True)
            # print(f"dr={np.linalg.norm(target - dr)}--->{np.linalg.norm(target - dr)**2}")
            return np.linalg.norm(target - dr)**2

    def local_constraints(s1):
        # print(f"s {s1}")
        # u = o.cases['repulse_vel_control'](s1[0:3])
        dr, e_max, V_max, R_max, w_max, j_max, g = calculation_motion(o=o, u=np.array(s1[0:3]), T_max=T_max, id_app=id_app,
                                                                      interaction=interaction, to_scipy=True)
        anw = [call_crash(o, i, o.r_ub, o.S, o.taken_beams, iFunc=True, brf=True) for i in g]
        # print(f"g {np.sum(np.array(anw) > 0) / len(anw)} | dr={np.linalg.norm(target - dr)}")
        if ifunc:
            tmp = dr - np.array(s1[3:6])
            anw += [tmp[0], -tmp[0], tmp[1], -tmp[1], tmp[2], -tmp[2]]
        # print(f"anw1 {anw}")
        # print(f"anw2 {[tmp[0], -tmp[0], tmp[1], -tmp[1], tmp[2], -tmp[2]]}")
        return anw

    res = scipy.optimize.minimize(local_func, s,  # args=(o, T_max, id_app, interaction, False, o.mu_ipm),
                                  tol=tol, method='trust-constr', options=opt,  # trust-constr SLSQP TNC
                                  bounds=bounds,
                                  constraints={'type': 'ineq', 'fun': local_constraints})
    print(f"res.x {res.x}")
    u = o.cases['repulse_vel_control'](res.x[0:3])
    return u
    
def local_get_hkw(r, v, t_, w_):
    C = [2 * r[2] + v[0] / w_, v[2] / w_, -3 * r[2] - 2 * v[0] / w_, r[0] - 2 * v[2] / w_, v[1] / w_, r[1]]
    return [-3 * C[0] * w_ * t_ + 2 * C[1] * np.cos(w_ * t_) - 2 * C[2] * np.sin(w_ * t_) + C[3],
                     C[5] * np.cos(w_ * t_) + C[4] * np.sin(w_ * t_),
                     2 * C[0] + C[2] * np.cos(w_ * t_) + C[1] * np.sin(w_ * t_)]

def approx_from_2d(t, M, m, R_x_0, R_z_0, phi_0, w_0, Vp_x, Vp_z, r_x_0, r_z_0, Rp_x, Rp_z, wp_y, xp_c, zp_c, x_c,
                   z_c, x_0, z_0, J_y, Jp_y, r1_x, r1_z):
    from numpy import sin, cos, sqrt
    return np.array([float(M*(2*R_x_0*w_0*sin(phi_0) - R_x_0*w_0*sin(phi_0 - t*w_0)/2 - 3*R_x_0*w_0*sin(phi_0 + t*w_0)/2 - 3*R_z_0*t*w_0**2*sin(phi_0 - t*w_0)/2 + 9*R_z_0*t*w_0**2*sin(phi_0 + t*w_0)/2 - 14*R_z_0*w_0*cos(phi_0) + 5*R_z_0*w_0*cos(phi_0 - t*w_0) + 9*R_z_0*w_0*cos(phi_0 + t*w_0) - 3*Vp_z*t*w_0*cos(phi_0 - t*w_0) + 3*Vp_z*t*w_0*cos(phi_0 + t*w_0) + 16*Vp_z*sin(phi_0) - 8*Vp_z*sin(phi_0 - t*w_0) - 8*Vp_z*sin(phi_0 + t*w_0) - 2*r_x_0*w_0*sin(phi_0) + r_x_0*w_0*sin(phi_0 - t*w_0)/2 + 3*r_x_0*w_0*sin(phi_0 + t*w_0)/2 + 3*r_z_0*t*w_0**2*sin(phi_0 - t*w_0)/2 - 9*r_z_0*t*w_0**2*sin(phi_0 + t*w_0)/2 + 14*r_z_0*w_0*cos(phi_0) - 5*r_z_0*w_0*cos(phi_0 - t*w_0) - 9*r_z_0*w_0*cos(phi_0 + t*w_0) + 3*t*w_0*wp_y*x_c*cos(2*phi_0 - t*w_0)/2 - 3*t*w_0*wp_y*x_c*cos(2*phi_0 + t*w_0)/2 - 3*t*w_0*wp_y*xp_c*cos(2*phi_0 - t*w_0)/2 + 3*t*w_0*wp_y*xp_c*cos(2*phi_0 + t*w_0)/2 - 3*t*w_0*wp_y*z_0*sin(t*w_0) - 3*t*w_0*wp_y*z_c*sin(2*phi_0 - t*w_0)/2 + 3*t*w_0*wp_y*z_c*sin(2*phi_0 + t*w_0)/2 + 3*t*w_0*wp_y*zp_c*sin(t*w_0) + 3*t*w_0*wp_y*zp_c*sin(2*phi_0 - t*w_0)/2 - 3*t*w_0*wp_y*zp_c*sin(2*phi_0 + t*w_0)/2 + 2*w_0*sqrt(2*r1_x**2 + 2*r1_z**2)*sin(phi_0) - w_0*sqrt(2*r1_x**2 + 2*r1_z**2)*sin(phi_0 - t*w_0)/2 - 3*w_0*sqrt(2*r1_x**2 + 2*r1_z**2)*sin(phi_0 + t*w_0)/2 - 8*wp_y*x_c*sin(2*phi_0) + 4*wp_y*x_c*sin(2*phi_0 - t*w_0) + 4*wp_y*x_c*sin(2*phi_0 + t*w_0) + 8*wp_y*xp_c*sin(2*phi_0) - 4*wp_y*xp_c*sin(2*phi_0 - t*w_0) - 4*wp_y*xp_c*sin(2*phi_0 + t*w_0) - 8*wp_y*z_0*cos(t*w_0) + 8*wp_y*z_0 - 8*wp_y*z_c*cos(2*phi_0) + 4*wp_y*z_c*cos(2*phi_0 - t*w_0) + 4*wp_y*z_c*cos(2*phi_0 + t*w_0) + 8*wp_y*zp_c*cos(2*phi_0) + 8*wp_y*zp_c*cos(t*w_0) - 4*wp_y*zp_c*cos(2*phi_0 - t*w_0) - 4*wp_y*zp_c*cos(2*phi_0 + t*w_0) - 8*wp_y*zp_c)/((M + m)*(3*t*w_0*sin(t*w_0) + 8*cos(t*w_0) - 8))),
             0, float(M*(sqrt(2)*w_0*sqrt(r1_x**2 + r1_z**2)*(-4*cos(phi_0) + cos(phi_0 - t*w_0) + 3*cos(phi_0 + t*w_0))/2 + (-4*cos(phi_0) + cos(phi_0 - t*w_0) + 3*cos(phi_0 + t*w_0))*(R_x_0*w_0 - 6*R_z_0*t*w_0**2 + 6*R_z_0*w_0*sin(t*w_0) - 4*Vp_z*cos(t*w_0) + 4*Vp_z - r_x_0*w_0 + 6*r_z_0*t*w_0**2 - 6*r_z_0*w_0*sin(t*w_0) + 3*t*w_0*wp_y*x_0*sin(phi_0) - 3*t*w_0*wp_y*x_c*sin(phi_0) + 3*t*w_0*wp_y*z_0*cos(phi_0) - 3*t*w_0*wp_y*z_c*cos(phi_0) - 4*wp_y*x_0*sin(phi_0)*sin(t*w_0) + 2*wp_y*x_0*cos(phi_0)*cos(t*w_0) - 2*wp_y*x_0*cos(phi_0) + 4*wp_y*x_c*sin(phi_0)*sin(t*w_0) + 2*wp_y*x_c*cos(phi_0)*cos(t*w_0) - 2*wp_y*x_c*cos(phi_0) - 4*wp_y*xp_c*cos(phi_0)*cos(t*w_0) + 4*wp_y*xp_c*cos(phi_0) - 2*wp_y*z_0*sin(phi_0)*cos(t*w_0) + 2*wp_y*z_0*sin(phi_0) - 4*wp_y*z_0*sin(t*w_0)*cos(phi_0) - 2*wp_y*z_c*sin(phi_0)*cos(t*w_0) + 2*wp_y*z_c*sin(phi_0) + 4*wp_y*z_c*sin(t*w_0)*cos(phi_0) + 4*wp_y*zp_c*sin(phi_0)*cos(t*w_0) - 4*wp_y*zp_c*sin(phi_0))/2 - (-3*t*w_0*cos(phi_0) - 2*sin(phi_0) - sin(phi_0 - t*w_0) + 3*sin(phi_0 + t*w_0))*(3*R_z_0*w_0*cos(t*w_0) - 4*R_z_0*w_0 + 2*Vp_z*sin(t*w_0) - 3*r_z_0*w_0*cos(t*w_0) + 4*r_z_0*w_0 - 2*wp_y*x_0*sin(phi_0)*cos(t*w_0) + 2*wp_y*x_0*sin(phi_0) - wp_y*x_0*sin(t*w_0)*cos(phi_0) + 2*wp_y*x_c*sin(phi_0)*cos(t*w_0) - 2*wp_y*x_c*sin(phi_0) - wp_y*x_c*sin(t*w_0)*cos(phi_0) + 2*wp_y*xp_c*sin(t*w_0)*cos(phi_0) + wp_y*z_0*sin(phi_0)*sin(t*w_0) - 2*wp_y*z_0*cos(phi_0)*cos(t*w_0) + 2*wp_y*z_0*cos(phi_0) + wp_y*z_c*sin(phi_0)*sin(t*w_0) + 2*wp_y*z_c*cos(phi_0)*cos(t*w_0) - 2*wp_y*z_c*cos(phi_0) - 2*wp_y*zp_c*sin(phi_0)*sin(t*w_0)))/((M + m)*(-3*t*w_0*sin(t*w_0) - 8*cos(t*w_0) + 8)))])


def repulsion(o, id_app, u_a_priori=None):
    """Функция отталкивания сборочного аппарата от конструкции."""
    """Input:                                                                   \n
    -> o - AllObjects класс                                                     \n
    -> id_app - номер аппарата                                                  \n
    -> u_a_priori - заданный вектор скорости отталкивания"""
    # Параметры до отталкивания
    N = 20
    o.repulse_app_config(id_app=id_app)
    r_1 = o.a.target[id_app]

    method_comps = o.method.split('+')
    target_is_reached = False
    if u_a_priori is not None:
        u0 = u_a_priori
        target_is_reached = True
    else:
        if 'diffevolve' in method_comps:
            u0, _ = diff_evolve(f_to_detour, [o.u_min, o.u_max], True, o, o.T_max, id_app, True, False, True,
                                n_vec=o.diff_evolve_vectors, chance=0.5, f=0.8, len_vec=3, n_times=o.diff_evolve_times,
                                multiprocessing=True, print_process=True)
        elif '2d_analytics' in method_comps:
            m_extra, M_without, J, J_1, J_p, r_center, r_center_p, r, R, r0c, R0c, R_p, V_p = \
                o.get_repulsion_change_params(id_app)
            r1 = o.b_o(r_1)
            r0_o = np.array(o.a.r[id_app])
            r0_b = np.array(o.a.target_p[id_app])
            count = 0
            phi_0 = 0
            phi_0 = my_atan2((o.S[0][0] + o.S[2][2])/2, (o.S[0][2] - o.S[2][0])/2)
            print(f"phi_0 = {phi_0}")
            for T in np.linspace(3000, o.T_max, N):
                u0 = approx_from_2d(Vp_x=V_p[0], Vp_z=V_p[2], Rp_x=R_p[0], Rp_z=R_p[2], r_x_0=r[0],
                                    r_z_0=r[2], phi_0=phi_0, wp_y=o.w[1], xp_c=r_center_p[0], zp_c=r_center_p[2],
                                    x_c=r_center[0], z_c=r_center[2], w_0=o.w_hkw,
                                    x_0=r0_b[0], z_0=r0_b[1], J_y=J[1][1], Jp_y=J_p[1][1], R_x_0=R[0],
                                    R_z_0=R[2], m=m_extra, M=M_without, t=T, r1_x=r_1[0], r1_z=r_1[2])
                # u0 = o.S @ u0
                u0 = np.array([u0[0], u0[2], u0[1]])
                count += 1
                o.my_print(f"Подбор {count}/{N} | u={u0}", mode='m')
                u1, target_is_reached = calc_shooting(o=o, id_app=id_app, r_1=r_1, interaction=True, u0=u0,
                                                      T_max=T + 50, func=f_to_capturing)
                if target_is_reached:
                    u0 = u1.copy()
                    break
        elif 'hkw_analytics' in method_comps:
            u0 = np.zeros(3)
            count = 0
            for T in np.linspace(3000, o.T_max, N):
                count += 1

                if count < 5 or count > 10:  # ШАМАНСТВО
                    flag_in_sphere = False
                    r1 = o.b_o(r_1)
                    r0 = np.array(o.a.r[id_app])
                    center = r0/2 + r1/2
                    radius = np.linalg.norm(r1 - r0) / 2
                    u = get_v0(o, id_app, T)
                    u0 = o.S @ u
                    C = get_c_hkw(r0, u, o.w_hkw)
                    o.my_print(f"Подбор {count}/{N} | u={u0}", mode='m')
                    for i_tmp in np.linspace(0, T, 30):  # Гиперпараметр 30
                        tmp = r_hkw(C, o.w_hkw, i_tmp)
                        # print(f"{round(np.linalg.norm(tmp - center) / radius * 100)}%")
                        if np.linalg.norm(tmp - center) < radius * 0.99:
                            print(f"Траектория внутри сферы!")
                            flag_in_sphere = True
                            break
                    if not flag_in_sphere:
                        u1, target_is_reached = calc_shooting(o=o, id_app=id_app, r_1=r_1, interaction=True, u0=u0,
                                                              T_max=T + 50, func=f_to_capturing)
                        # u1 = find_repulsion_velocity_new(o, id_app, r_1, True, u0, T, ifunc=False)
                        # u1 = my_calc_shooting(o, id_app, r_1, True, u=np.append([T], u0), func=f_dr)
                        '''dr, _, _, _, _, _, _, _, _, _ = calculation_motion(o=o, u=u0, T_max=T+200, id_app=id_app,
                                                                           interaction=True, check_visible=False)
                        target_is_reached = np.linalg.norm(dr) < o.d_to_grab
                        o.my_print(f"Итоговая точность {np.linalg.norm(dr)}", mode='m')'''
                        if target_is_reached:
                            u0 = u1.copy()
                            break
                # u0 = o.S @get_v0(o, id_app, o.T_max)
                # u0 = np.array([-0.01692162423579614, 5.20798226662381e-05, 0.008768796686505929])
                # print(f"Аналитическая скорость {u0}")
                # u0 = o.S @ u0
        else:
            tmp = r_1 - np.array(o.o_b(o.a.r[id_app]))
            u0 = o.u_min * tmp / np.linalg.norm(tmp)
            u0 = np.array([-0.00163023, -0.00777977,  0.0007209])
        '''if not target_is_reached:
            u1 = find_repulsion_velocity_new(o, id_app, r_1, True, u0, T, ifunc=False)
            u1, target_is_reached = my_calc_shooting(o, id_app, r_1, True, u=np.append([T], u1), func=f_dr)
            if target_is_reached:
                u0 = u1.copy()'''

        # Доводка
        if 'shooting' in method_comps:
            u1, target_is_reached = calc_shooting(o=o, id_app=id_app, r_1=r_1, interaction=True, u0=u0)
            if target_is_reached:
                u0 = u1
        elif 'const-propulsion' in method_comps:
            v0, _ = diff_evolve(f_controlled_const, [o.u_min, o.u_max, 0, o.a_pid_max / 1.7], True, o, o.T_max, id_app,
                                True, False, True, n_vec=o.diff_evolve_vectors, chance=0.5, f=0.8, len_vec=6,
                                n_times=o.diff_evolve_times, multiprocessing=True, print_process=True)
            v0 = calc_shooting(o=o, id_app=id_app, r_1=r_1, interaction=True, func=f_controlled_const, n=6, u0=v0)
            u0 = v0[0:3]
            o.a_self[id_app] = o.cases['acceleration_control'](v0[3:6])
            print(f"TEPER {v0}")
        elif 'linear-propulsion' in method_comps:
            LEN_VEC = 9
            '''v0, _ = diff_evolve(f_controlled_const, [o.u_min, o.u_max, 0, o.a_pid_max / 1.7], True, o, o.T_max, id_app,
                                True, False, True, n_vec=o.diff_evolve_vectors, chance=0.5, f=0.8, len_vec=LEN_VEC,
                                n_times=o.diff_evolve_times, multiprocessing=True, print_process=True)'''
            v0 = np.array([0.022322849870837592, -0.0, -0.024082491687662064, 0., 0., 0., 0., 0., 0.])
            # v0 = np.array([1.57834221e-03, 6.17833125e-03, 1.13950052e-02, 2.56554088e-06, 1.74561577e-05, 4.72087501e-05, 2.90559321e-05, 2.49372057e-05, 4.00058404e-05])
            # v0 = np.append(v0, np.zeros(3))
            v0 = calc_shooting(o=o, id_app=id_app, r_1=r_1, interaction=True, func=f_controlled_const, n=LEN_VEC, u0=v0)
            ## v0 = np.array([-7.20616021e-05, -7.05597190e-03, -1.14687768e-02, -6.43142795e-07, 7.01282186e-07, -5.12799953e-07,  1.49374925e-06,  1.14988169e-06, -2.00274023e-06])
            # v0 = np.array([2.51244937e-03, 5.30263254e-03, 9.29356095e-03, 3.70858621e-05, 5.96174803e-06, 5.37664468e-05, 3.65594966e-05, 1.72421048e-05, 4.68459118e-05])
            print(len(v0))
            u0 = v0[0:3]
            o.a_self_params[id_app] = v0[3:LEN_VEC]
            print(f"TEPER {v0}")
        elif 'linear-angle' in method_comps:
            LEN_VEC = 7
            v0, _ = diff_evolve(f_controlled_const, [o.u_min, o.u_max, 0, 2*np.pi], True, o, o.T_max, id_app,
                                True, False, True, n_vec=o.diff_evolve_vectors, chance=0.5, f=0.8, len_vec=LEN_VEC,
                                n_times=o.diff_evolve_times, multiprocessing=True, print_process=True)
            v0 = calc_shooting(o=o, id_app=id_app, r_1=r_1, interaction=True, func=f_controlled_const, n=LEN_VEC, u0=v0)
            u0 = v0[0:3]
            o.a_self_params[id_app] = v0[3:LEN_VEC]
        elif 'trust-constr' in method_comps:
            u0 = find_repulsion_velocity(o=o, id_app=id_app, target=r_1, interaction=True)
    if 'linear-propulsion' in method_comps or 'const-propulsion' in method_comps:
        o.a.flag_hkw[id_app] = False
    if not target_is_reached and 'pd' in method_comps:
        o.my_print('Я не справляюсь, включаю ПД', mode="test")
        o.control = True
        o.if_PID_control = True
    if not target_is_reached and 'imp' in method_comps:
        o.my_print('Я не справляюсь, включаю импульсы', mode="test")
        o.control = True
        o.if_impulse_control = True

    u0 = o.cases['repulse_vel_control'](u0)
    if target_is_reached:
        o.repulsion_change_params(id_app=id_app, u0=u0)
        o.my_print(f"КА id={id_app} оттолкнулся со скоростью u={np.linalg.norm(u0)}, w={np.linalg.norm(o.w)}",
                   mode="b", test=True)
        o.my_print(f"Взятые стержни: {o.taken_beams}", mode="g", test=True)
        talk_decision(o.if_talk)
    else:
        o.remove_repulse_app_config(id_app)
    return u0


def capturing(o, id_app):
    if o.method == 'shooting+pd':
        o.control = False
        o.if_PID_control = False
    if o.method == 'shooting+imp':
        o.control = False
        o.if_impulse_control = False

    if o.if_any_print:
        print(Fore.BLUE + f'Аппарат id:{id_app} захватился')
    o.a_self[id_app] = np.array(np.zeros(3))
    # talk_success(o.if_talk)
    o.capturing_change_params(id_app)
