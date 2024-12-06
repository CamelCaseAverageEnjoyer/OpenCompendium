import numpy as np

def simple_control(o, a, tau):
    from mylibs.tiny_functions import polar2dec, clip
    if len(a) == 3:
        return o.cases['acceleration_control'](a)
    elif len(a) == 4:
        return polar2dec(o.a_pid_max, clip(tau, 0, 1) * a[0] + clip(1 - tau, 0, 1) * a[2], clip(tau, 0, 1) * a[1] + clip(1 - tau, 0, 1) * a[3])
    elif len(a) == 6:
        return o.cases['acceleration_control'](clip(tau, 0, 1) * np.array(a[0:3]) +
                                               clip(1 - tau, 0, 1) * np.array(a[3:6]))
    elif len(a) == 9:
        if tau < 0.5:
            return o.cases['acceleration_control'](clip(2 * tau, 0, 1) * np.array(a[0:3]) +
                                                   clip(1 - 2 * tau, 0, 1) * np.array(a[3:6]))
        else:
            return o.cases['acceleration_control'](clip(2 * tau - 1, 0, 1) * np.array(a[3:6]) +
                                                   clip(2 - 2 * tau, 0, 1) * np.array(a[6:9]))

def kd_from_kp(k):
    return 2 * np.sqrt(k)

def force_from_beam(a, diam, n, tau, b, f0: float, f1: float, f2: float, k_av: float = 1e-5, level: int = 2):
    from mylibs.tiny_functions import clip
    """Возвращает в ССК!"""
    if (f0 > -1) and (f0 < 1):
        a1 = a - f0 * n / 2 if (f1**2 + f2**2 > 1) else np.zeros(3)
    else:
        a1 = a - np.sign(f0) * n / 2
    # tmp = k_av / np.sqrt(clip(np.linalg.norm(a1) - diam, 1e-8, 1e9999)) # ** level
    tmp = k_av / (clip(np.linalg.norm(a1) - diam, 1e-8, 1e9999)) ** level
    return a1 / np.linalg.norm(a1) * tmp


def avoiding_force(o, id_app, r=None):
    from mylibs.calculation_functions import call_crash_internal_func
    if r is None:
        r = o.o_b(o.a.r[id_app])
    else:
        r = o.o_b(r)
    force = np.zeros(3)

    for i in range(o.s.n_beams):
        if not(np.any(o.taken_beams == i)):
            if np.sum(o.s.flag[i]) > 0:
                r1 = o.s.r1[i]
                r2 = o.s.r2[i]
            else:
                r1 = [o.s.r_st[i][0], o.s.r_st[i][1], o.s.r_st[i][2]]
                r2 = [o.s.r_st[i][0] - o.s.length[i], o.s.r_st[i][1], o.s.r_st[i][2]]
            tmp = call_crash_internal_func(r, r1, r2, o.d_crash, return_force=True, k_av=o.k_av, level=o.level_avoid)
            if tmp is not False:
                if o.a.n > 0:
                    if np.linalg.norm(tmp) > np.linalg.norm(force) and np.linalg.norm(o.a.target[id_app] - r1) > 0.5:
                        force = tmp.copy()
                else:
                    if np.linalg.norm(tmp) > np.linalg.norm(force):
                        force = tmp.copy()

    for i in range(o.c.n):
        r1 = o.c.r1[i]
        r2 = o.c.r2[i]
        tmp_point = (np.array(r1) + np.array(r2)) / 2
        tmp_force = call_crash_internal_func(r, r1, r2, o.c.diam[i], return_force=True, k_av=o.k_av, level=o.level_avoid)
        if tmp_force is not False:
            if o.a.n > 0:
                if np.linalg.norm(tmp_force) > np.linalg.norm(force) and np.linalg.norm(o.a.target[id_app] - tmp_point) > 0.5:
                    force = tmp_force.copy()
            else:
                if np.linalg.norm(tmp_force) > np.linalg.norm(force):
                    force = tmp_force.copy()
    return o.S.T @ force

def pd_control(o, id_app):
    o.a.flag_hkw[id_app] = False
    r = np.array(o.a.r[id_app])
    v = np.array(o.a.v[id_app])
    dr = o.get_discrepancy(id_app, vector=True)
    dv = (dr - o.dr_p[id_app]) / o.dt
    o.dr_p[id_app] = dr.copy()
    r1 = o.a.target[id_app] - o.r_center
    a_pid = -o.k_p * dr - o.k_d * dv  # \
    '''         + o.S.T @ (my_cross(o.S @ o.e, r1) + my_cross(o.S @ o.w, my_cross(o.S @ o.w, r1)) +
                        2 * my_cross(o.S @ o.w, o.S @ o.a.v[id_app])) \
             + o.A_orbital - o.a_orbital[id_app]'''
    o.a_self[id_app] = a_pid.copy()

def lqr_control(o, id_app):
    import scipy
    from mylibs.tiny_functions import my_cross, clip

    o.a.flag_hkw[id_app] = False
    r = np.array(o.a.r[id_app])
    v = np.array(o.a.v[id_app])
    rv = np.append(r, v)
    r1 = o.a.target[id_app] - o.r_center
    muRe = o.mu / o.Radius_orbit ** 3
    tmp = 3 / o.Radius_orbit ** 4
    '''a = np.array([[0, 0, 0, 1., 0, 0],
                  [0, 0, 0, 0, 1., 0],
                  [0, 0, 0, 0, 0, 1.],
                  [(-muRe*o.A[0][0] + o.Om[1]**2 + o.Om[2]**2 + tmp*o.R_e[0]*o.S[0][2]),
                   (-muRe*o.A[1][0] + o.Om[2] - o.Om[0]*o.Om[1] + tmp*o.R_e[0]*o.S[1][2]),
                   (-muRe*o.A[2][0] - o.Om[1] - o.Om[0]*o.Om[2] + tmp*o.R_e[0]*o.S[2][2]),
                   0, 2*o.Om[2], -2*o.Om[1]],
                  [(-muRe*o.A[0][1] - o.Om[2] - o.Om[1]*o.Om[0] + tmp*o.R_e[1]*o.S[0][2]),
                   (-muRe*o.A[1][1] + o.Om[0]**2 + o.Om[2]**2 + tmp*o.R_e[1]*o.S[1][2]),
                   (-muRe*o.A[2][1] + o.Om[0] - o.Om[1]*o.Om[2] + tmp*o.R_e[1]*o.S[2][2]),
                   -2*o.Om[2], 0, 2*o.Om[0]],
                  [(-muRe*o.A[0][2] + o.Om[1] - o.Om[2]*o.Om[0] + tmp*o.R_e[2]*o.S[0][2]),
                   (-muRe*o.A[1][2] - o.Om[0] - o.Om[2]*o.Om[1] + tmp*o.R_e[2]*o.S[1][2]),
                   (-muRe*o.A[2][2] + o.Om[0]**2 + o.Om[1]**2 + tmp*o.R_e[2]*o.S[2][2]),
                   2*o.Om[1], -2*o.Om[0], 0]])'''
    a = np.array([[0, 0, 0, 1., 0, 0],
                  [0, 0, 0, 0, 1., 0],
                  [0, 0, 0, 0, 0, 1.],
                  [o.Om[1] ** 2 + o.Om[2] ** 2, o.e[2] - o.Om[0] * o.Om[1], -o.e[1] - o.Om[0] * o.Om[2],
                   0, 2 * (o.Om[2] - o.w_hkw_vec[2]), -2 * o.w_hkw - 2 * (o.Om[1] - o.w_hkw_vec[1])],
                  [-o.e[2] - o.Om[0] * o.Om[1], o.Om[0] ** 2 + o.Om[2] ** 2 - o.w_hkw ** 2, o.e[0] - o.Om[1] * o.Om[2],
                   -2 * (o.Om[2] - o.w_hkw_vec[2]), 0, 2 * (o.Om[0] - o.w_hkw_vec[0])],
                  [o.e[1] - o.Om[0] * o.Om[2], -o.e[0] - o.Om[1] * o.Om[2], o.Om[0] ** 2 + o.Om[1] ** 2 + 3 * o.w_hkw ** 2,
                   2 * o.w_hkw + 2 * (o.Om[1] - o.w_hkw_vec[1]), -2 * (o.Om[0] - o.w_hkw_vec[0]), 0]])
    # print(f"a:{a}")
    b = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [1., 0, 0],
                  [0, 1., 0],
                  [0, 0, 1.]])
    x_rate = 1
    u_rate = 1e9
    q = np.eye(6) * x_rate
    r = np.eye(3) * u_rate
    p = scipy.linalg.solve_continuous_are(a, b, q, r)

    # print(f"k:{np.linalg.inv(r) @ b.T @ p}")
    a_lqr = - np.linalg.inv(r) @ b.T @ p @ rv
    # a_lqr += tmp * o.R_e * (o.R[2] - (o.S.T @ o.a.target[id_app])[2]) - tmp * o.R_e * o.R[2] - tmp * o.U.T @ o.R * o.R[2] + \
    #          tmp * o.U.T @ o.a.r[id_app] + muRe * o.A.T @ o.a.target[id_app]
    a_lqr += my_cross(o.w_hkw_vec, my_cross(o.w_hkw_vec, o.r_ub)) + \
             o.get_hkw_acceleration(np.append(o.S.T @ (o.a.target[id_app] - o.r_center), o.v_ub + my_cross(o.w, o.a.target[id_app] - o.r_center)))
    print(f"проверка: {np.linalg.norm(a_lqr) / o.a_pid_max * 100}%")
    '''a_lqr = - np.linalg.inv(r) @ b.T @ p @ rv \
            + o.S.T @ (my_cross(o.S @ o.e, r1) + my_cross(o.S @ o.w, my_cross(o.S @ o.w, r1)) +
                       2 * my_cross(o.S @ o.w, o.S @ o.a.v[id_app])) \
            + o.A_orbital - o.a_orbital[id_app]'''
    # a_lqr -= o.a_self[id_app]
    a_lqr *= clip(np.linalg.norm(a_lqr), 0, o.a_pid_max) / np.linalg.norm(a_lqr)
    o.a_self[id_app] = a_lqr.copy()

def impulse_control(o, id_app):
    from mylibs.tiny_functions import get_c_hkw
    from mylibs.calculation_functions import diff_evolve, f_to_detour, talk_flyby, calc_shooting, talk_shoot

    if not o.flag_vision[id_app]:
        o.t_flyby_counter -= o.dt
        o.t_reaction_counter = o.t_flyby
        if o.t_flyby_counter < 0:  # Облёт конструкции
            o.my_print('Облёт', mode='r')
            # u = find_repulsion_velocity(o=o, id_app=id_app, interaction=False)
            u, _ = diff_evolve(f_to_detour, [[o.u_min, o.u_max] for _ in range(3)], o, o.T_max, id_app, False, False,
                               n_vec=o.diff_evolve_vectors, chance=0.5, f=0.8, len_vec=3, n_times=5, multiprocessing=True, print_process=True)
            o.t_flyby_counter = o.t_flyby
            o.t_start[id_app] = o.t
            talk_flyby(o.if_talk)
            o.C_r[id_app] = get_c_hkw(o.a.r[id_app], u, o.w_hkw)
    else:
        o.t_flyby_counter = o.t_flyby
        o.t_reaction_counter -= o.dt
        if o.t_reaction_counter < 0:  # Точное попадание в цель
            o.my_print('Попадание', mode='r')
            r_right = o.b_o(o.a.target[id_app])
            # u, _ = diff_evolve(f_to_detour, [[o.u_min, o.u_max] for _ in range(3)], o, o.T_max, id_app, False, False,
            #                    n_vec=8, chance=0.5, f=0.8, len_vec=3, n_times=5, multiprocessing=True, print_process=True)
            u, target_is_reached = calc_shooting(o=o, id_app=id_app, r_right=r_right, interaction=False)
            o.t_reaction_counter = o.t_reaction
            o.t_start[id_app] = o.t
            talk_shoot(o.if_talk)
            o.flag_impulse = not target_is_reached
            o.C_r[id_app] = get_c_hkw(o.a.r[id_app], u, o.w_hkw)

def control_condition(o, id_app, return_percentage=False):
    from mylibs.calculation_functions import call_crash
    target_orf = o.b_o(o.a.target[id_app])
    see_rate = 1
    not_see_rate = 10
    points = 100
    crash_points = points
    if o.flag_vision[id_app]:  # If app have seen target, then app see it due to episode end
        o.t_reaction_counter -= o.dt
        return True, 0 if return_percentage else True
    if o.a.flag_fly[id_app] and ((o.flag_vision[id_app] and ((o.iter % see_rate) == 0)) or
                                 ((not o.flag_vision[id_app]) and ((o.iter % not_see_rate) == 0))):
        o.flag_vision[id_app] = True
        for j in range(points):
            intermediate = (target_orf * j + np.array(o.a.r[id_app]) * (points - j)) / points
            if call_crash(o, intermediate, o.r_ub, o.S, o.taken_beams):
                o.flag_vision[id_app] = False 
                crash_points = j
                break
        o.t_reaction_counter = o.t_reaction_counter - o.dt*see_rate if o.flag_vision[id_app] else o.t_reaction
        return o.flag_vision[id_app], (points - crash_points) / points if return_percentage else o.flag_vision[id_app]
    '''else:
        return True'''
    return False,  (points - crash_points) / points if return_percentage else False
