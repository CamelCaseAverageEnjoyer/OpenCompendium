import matplotlib.pyplot as plt

from all_objects import *
from vedo import *
from datetime import datetime
k_ac_list = [0.01, 0.02, 0.05, 0.1, 0.2]
FONTSIZE = 10
CONSTRUCTION_LIST = ['2', '3']


def reader_pd_control_params(name: str = '', eps: float = 1e-1, lng: str = 'ru'):
    global k_ac_list
    filename = 'storage/pid_koeff_' + name + '.txt'
    f = open(filename, 'r')
    k_p, k_a, tol, col = ([], [], [], [])
    box_k_p = [[] for _ in range(5)]
    for line in f:
        lst = line.split()
        k_p += [float(lst[0])]
        k_a += [int(lst[1])]
        tol += [float(lst[2])]
        col += [int(lst[3])]
    f.close()

    tol_line = [[] for _ in range(len(k_ac_list))]
    count = [0 for _ in range(len(k_ac_list))]
    k_p_max = 0.
    k_p_list = []
    for i in range(len(tol)):
        if k_p_max < k_p[i]:
            k_p_max = k_p[i]
            k_p_list += [k_p[i]]
            for k in range(len(k_ac_list)):
                if count[k_a[i]] > 0:
                    tol_line[k][len(k_p_list) - 2] /= count[k]
                tol_line[k].append(0.)
            count = [0 for _ in range(len(k_ac_list))]
        else:
            tol_line[k_a[i]][len(k_p_list) - 1] += tol[i]
            count[k_a[i]] += 1
    for k in range(len(k_ac_list)):
        tol_line[k][len(k_p_list) - 1] /= count[k]

    if lng == 'ru':
        title = "Подбор коэффициентов ПД-регулятора"
        x_label = "Коэффициент k_p"
        y_label = "Точность"
    else:
        title = "PD-controller coeffitient enumeration"
        x_label = "k_p coeffitient"
        y_label = "Tolerance"

    clr = [['aqua', 'violet', 'limegreen', 'gold', 'lightcoral'],
           ['steelblue', 'purple', 'green', 'goldenrod', 'firebrick']]
    plt.title(title)
    for i in range(len(k_ac_list)):
        plt.plot(k_p_list, tol_line[i], c=clr[0][i], label=k_ac_list[i])
        for kp in k_p_list:
            plt.plot([kp * (1 + eps * (i - 2))] * 2, [0., 20.], c=clr[0][i])
    for i in range(len(tol)):
        plt.scatter(k_p[i] * (1 + eps * (k_a[i] - 2)), tol[i], c=clr[col[i]][k_a[i]])
    plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def pd_control_params_search(name: str = '', dt=0.2, n_p=5, n_a=10, T_max=700., k_min=1e-4, k_max=1e-2):
    """Функция ищет смысл жизни(его нет) / зависимость невязки/столкновения от """
    global k_ac_list
    k_p_list = np.exp(np.linspace(np.log(k_min), np.log(k_max), n_p))  # Logarithmic scale
    k_p_best = 0
    tolerance_best = 1e5

    filename = 'storage/pid_koeff_' + name + '.txt'
    f = open(filename, 'w')
    start_time = datetime.now()
    tmp_count = 0
    collide = False

    for k_p in k_p_list:
        tol_count = 0.
        for k_a_i in range(len(k_ac_list)):
            for _ in range(n_a):
                tmp_count += 1
                id_app = 0
                tolerance = None
                print(Fore.CYAN + f'Подбор ПД-к-в: {tmp_count}/{n_p * n_a * len(k_ac_list)};'
                                  f' время={datetime.now() - start_time}' + Style.RESET_ALL)
                o = AllProblemObjects(if_PID_control=True,
                                      dt=dt, k_p=k_p, k_ac=k_ac_list[k_a_i],
                                      T_max=T_max,
                                      if_talk=False,
                                      if_any_print=False,
                                      choice='3')

                for i_time in range(int(T_max // dt)):
                    # Repulsion
                    o.a.busy_time[id_app] -= o.dt if o.a.busy_time[id_app] >= 0 else 0
                    if o.a.flag_fly[id_app] == 0 and o.a.busy_time[id_app] < 0:
                        _ = repulsion(o, id_app, u_a_priori=np.array([-0.00749797, 0.00605292, 0.08625441]))

                    o.time_step()
                    o.control_step(id_app)

                    tmp = o.get_discrepancy(id_app)
                    collide = call_crash(o, o.a.r[id_app], o.r_ub, o.S, o.taken_beams)
                    if tolerance is None or tolerance > tmp:
                        tolerance = tmp
                    if collide:
                        break
                tol_count += tolerance
                f.write(f'{k_p} {k_a_i} {tolerance} {int(collide)}\n')
        tol_count /= n_a * len(k_ac_list)
        if tol_count < tolerance_best:
            tolerance_best = tol_count
            k_p_best = k_p
    f.close()
    reader_pd_control_params(name=name)
    return k_p_best

def get_repulsions(filename: str = ""):
    f = open('storage/main' + filename + '.txt', 'r')
    for line in f:
        lst = line.split()
        if lst[0] == 'отталкивание':
            print(f"Отталкивание {lst[1]}: [{lst[2]}, {lst[3]}, {lst[4]}]")
    f.close()

def plot_params_while_main(filename: str = "", trial_episodes: bool = False, show_rate: int = 1, limit: int = 1e5,
                           show_probe_episodes=True, dt: float = 1., energy_show=True, t_from=0., show_w=True,
                           show_j=True, show_V=True, show_R=True, propulsion_3d_plot: bool = True):
    f = open('storage/main' + filename + '.txt', 'r')
    o = AllProblemObjects()

    id_max = 0
    for line in f:
        lst = line.split()
        if lst[0] == 'график':
            id_max = max(id_max, 1 + int(lst[1]))
    f.close()

    def params_reset():
        return [[[] for _ in range(id_max)] for _ in range(11)]

    dr, e, j, V, R, t, a, m, mc, ub, r = params_reset()  # mc - mass center
    R_max, V_max, j_max, e_max = (1., 1., 1., 1.)

    f = open('storage/main' + filename + '.txt', 'r')
    tmp = 0
    for line in f:
        lst = line.split()
        if len(dr[0]) < limit:
            if len(lst) > 0:
                # print(len(lst))
                if lst[0] == 'ограничения' and len(lst) == 5:
                    R_max = float(lst[1])
                    V_max = float(lst[2])
                    j_max = float(lst[3])
                    e_max = float(lst[4])
                    o.my_print(f"Есть ограничения: r_ub={R_max}, v_ub={V_max}, j_ub={j_max}, w_ub={e_max}", test=True)
                if lst[0] == 'график' and tmp * dt > t_from and tmp % show_rate == 0:
                    if len(lst) == 9 and (show_probe_episodes or bool(int(lst[8]))):
                        id_app = int(lst[1])
                        dr[id_app].append(float(lst[2]))
                        e[id_app].append(float(lst[3]))
                        j[id_app].append(float(lst[4]))
                        V[id_app].append(float(lst[5]))
                        R[id_app].append(float(lst[6]))
                        a[id_app].append(float(lst[7]))
                        m[id_app].append(int(lst[8]))
                    if len(lst) == 11 and (show_probe_episodes or bool(int(lst[10]))):
                        id_app = int(lst[1])
                        dr[id_app].append(float(lst[2]))
                        e[id_app].append(float(lst[3]))
                        j[id_app].append(float(lst[4]))
                        V[id_app].append(float(lst[5]))
                        R[id_app].append(float(lst[6]))
                        a[id_app].append(float(lst[7]))
                        mc[id_app].append(float(lst[8]))
                        r[id_app].append(float(lst[9]))

            if lst[0] == 'отталкивание':
                print(f"Отталкивание {lst[1]}: [{lst[2]}, {lst[3]}, {lst[4]}]")
        else:
            print(Fore.MAGENTA + f"Внимание! Превышен лимит в {limit} точек!" + Style.RESET_ALL)
            break
        tmp += 1
    f.close()
    print(Fore.CYAN + f"Аппаратов на графиках: {id_max}" + Style.RESET_ALL)
    print(Fore.BLUE + f"Точек на графиах: {len(dr[0])}" + Style.RESET_ALL)

    p = 3 if propulsion_3d_plot else 2
    fig, axs = plt.subplots(p)
    axs[0].set_ylabel('Position error Δr(t), m', fontsize=11)
    axs[0].set_title('', fontsize=13)
    axs[1].set_xlabel('Time t, s', fontsize=11)
    axs[1].set_ylabel('Relative constrained variables', fontsize=11)
    if propulsion_3d_plot:
        axs[2].set_xlabel('Time t, s', fontsize=11)
        axs[2].set_ylabel('бортовое ускорение, м/с2', fontsize=11)

    clr = ['c', 'indigo', 'm', 'violet', 'teal', 'slategray', 'greenyellow', 'sienna']
    clr2 = [['skyblue', 'bisque', 'palegreen', 'darksalmon'], ['teal', 'tan', 'g', 'brown']]
    if show_probe_episodes:
        for id_app in range(id_max):
            t[id_app] = np.linspace(t_from, t_from + len(dr[id_app]) * dt * show_rate, len(dr[id_app])) + t_from
            for i in range(len(dr[id_app]) - 1):
                axs[0].plot([t[id_app][i], t[id_app][i+1]], np.array([dr[id_app][i], dr[id_app][i+1]]),
                            c=clr[2 * id_app + 2 * m[id_app][i]])
            axs[1].plot(t[id_app], [1 for _ in range(len(t[id_app]))], c='gray')
            axs[0].plot(t[id_app], np.zeros(len(t[id_app])), c='khaki')
            if propulsion_3d_plot:
                axs[2].plot(range(len(a[id_app])), a[id_app], c='c')
                axs[2].plot(range(len(a[id_app])), np.zeros(len(a[id_app])), c='khaki')
        id_app = 0
        clr = [['skyblue', 'bisque', 'palegreen', 'darksalmon'], ['teal', 'tan', 'g', 'brown']]
        if energy_show:
            axs[1].plot([t[id_app][0], t[id_app][1]], [np.array(e[id_app][0]) / e_max, np.array(e[id_app][1]) /
                                                    e_max], c=clr[1][0], label='энергия')
        '''axs[1].plot([t[id_app][0], t[id_app][1]], [np.array(j[id_app][0]) / j_max, np.array(j[id_app][1]) /
                                                   j_max], c=clr[1][1], label='угол')'''
        axs[1].plot([t[id_app][0], t[id_app][1]], [np.array(V[id_app][0]) / V_max, np.array(V[id_app][1]) /
                                                   V_max], c=clr[1][2], label='V')
        '''axs[1].plot([t[id_app][0], t[id_app][1]], [np.array(R[id_app][0]) / R_max, np.array(R[id_app][1]) /
                                                   R_max], c=clr[1][3], label='R')'''
        for i in range(len(t[id_app]) - 1):
            if energy_show:
                axs[1].plot([t[id_app][i], t[id_app][i+1]], [np.array(e[id_app][i]) / e_max, np.array(e[id_app][i+1]) /
                                                            e_max], c=clr[m[id_app][i]][0])
            '''axs[1].plot([t[id_app][i], t[id_app][i+1]], [np.array(j[id_app][i]) / j_max, np.array(j[id_app][i+1]) /
                                                         j_max], c=clr[m[id_app][i]][1])'''
            axs[1].plot([t[id_app][i], t[id_app][i+1]], [np.array(V[id_app][i]) / V_max, np.array(V[id_app][i+1]) /
                                                         V_max], c=clr[m[id_app][i]][2])
            '''axs[1].plot([t[id_app][i], t[id_app][i+1]], [np.array(R[id_app][i]) / R_max, np.array(R[id_app][i+1]) /
                                                         R_max], c=clr[m[id_app][i]][3])'''
    else:
        for id_app in range(id_max):
            t[id_app] = np.linspace(0, len(dr[id_app]), len(dr[id_app])) * dt * show_rate + t_from
            if energy_show:
                axs[0].plot(t[id_app], ub[id_app], c=clr[2 * id_app + 3], label='UB mass center')
                axs[0].plot(t[id_app], mc[id_app], c=clr[2 * id_app + 1], label='Total mass center')
            axs[0].plot(t[id_app], dr[id_app], c=clr[2 * id_app], label='Δr(t)')
            # axs[0].plot(t[id_app], mc[id_app], c=clr[2 * id_app + 2], label='mass center(t)')
            # axs[0].plot(t[id_app], r[id_app], c=clr[2 * id_app + 3], label='r(t)')
            # axs[0].plot(t[id_app], R[id_app], c=clr[2 * id_app + 4], label='R(t)')
            # axs[1].plot(t[id_app], [1 for _ in range(len(t[id_app]))], c='gray')  # Серая линия
            if propulsion_3d_plot:
                axs[2].plot(t[id_app], a[id_app], c='c')
            # axs[0].plot(t[id_app], np.zeros(len(t[id_app])), c='khaki')
            # axs[2].plot(range(len(a[id_app])), np.zeros(len(a[id_app])), c='khaki')
        id_app = 0
        if show_w:
            axs[1].plot(t[id_app], np.array(e[id_app]) / e_max, c=clr2[1][0], label='ωᵘᵇ')
        if show_j:
            axs[1].plot(t[id_app], np.array(j[id_app]) / j_max, c=clr2[1][1], label='ηᵘᵇ')
        if show_V:
            axs[1].plot(t[id_app], np.array(V[id_app]) / V_max, c=clr2[1][2], label='υᵘᵇ')
        if show_R:
            axs[1].plot(t[id_app], np.array(R[id_app]) / R_max, c=clr2[1][3], label='rᵘᵇ')
    axs[0].legend()
    axs[1].legend()
    axs[1].set_ylim([0, 1])
    plt.grid(True)
    plt.show()


def plot_a_avoid(x_boards: list = np.array([-10, 10]), z_boards: list = np.array([-5, 10])):
    o = AllProblemObjects(choice='0', N_apparatus=0)

    # x_boards: list = [-15, 15], z_boards: list = [1, 10]
    nx = 60
    nz = 60
    x_list = np.linspace(x_boards[0], x_boards[1], nx)
    z_list = np.linspace(z_boards[0], z_boards[1], nz)

    arrows = []
    i = 0
    forces = [np.zeros(3) for i in range(nx*nz)]
    max_force = 0
    for x in x_list:
        for z in z_list:
            tmp = avoiding_force(o, 0, r=[x, 0, z])
            if tmp is not False:
                forces[i] = tmp
                max_force = max(max_force, np.linalg.norm(tmp))
            i += 1
    i = 0
    for x in x_list:
        for z in z_list:
            force = forces[i] / max_force
            i += 1
            if force is not False:
                print(f"force: {force}")
                l1 = [np.array([x, 0, z]), np.array([x, 0, z]) + force]
                l2 = [np.array([x - 0.1*force[2], 0, z + 0.1*force[0]]),
                      np.array([x - 0.1*force[2], 0, z + 0.1*force[0]]) + force]
                farr = FlatArrow(l1, l2, tip_size=1, tip_width=1).c(color='c', alpha=0.9)
                arrows.append(farr)

    arrows.append(plot_iterations_new(o).color("silver"))
    show(arrows, __doc__, viewup="z", axes=1, bg='white', zoom=1, size=(1920, 1080)).close()

def reader_avoid_field_params_search(filename: str = '', lng: str = 'ru'):
    # Init
    f = open(filename, 'r')
    k_p, k_a, lvl, res = ([], [], [], [])
    for line in f:
        lst = line.split()
        k_p += [float(lst[0])]
        k_a += [float(lst[1])]
        lvl += [int(lst[2])]
        res += [int(lst[3])]
    f.close()

    # Titles
    if lng == 'ru':
        title = "Подбор коэффициентов ПД-регулятора, поля отталкивания"
        state_list = ['преследование', 'столкновение', 'попадание']
        xlabel = "коэффициент ПД-регулятора"
        ylabel = "коэффициент поля уклонения"
    else:
        title = "?"
        state_list = ['', 'collision', 'reaching']
        xlabel = "PD-controller coeffitient"
        ylabel = "avoiding field coeffitient"

    # Plotting
    clr = ['lightblue', 'deeppink', 'palegreen']
    flag = [True] * 3
    for i in range(len(res)):
        state = int(res[i] + 1)
        if flag[state]:
            plt.scatter(k_p[i], k_a[i], c=clr[state], label=state_list[state])
            flag[state] = False
        else:
            plt.scatter(k_p[i], k_a[i], c=clr[state])
    plt.title(title)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_avoid_field_params_search(name: str = '', dt=1.0, N=10, T_max=2000., k_p_min=1e-4, k_p_max=1e-2,
                                   k_a_min=1e-7, k_a_max=1e-3):
    """Фунция тестит коэффициенты ПД-регулятора и коэффициент отталкивания на конструкции 5.
    Результат - точки на пространстве {k_PD, k_avoid}, разделящиеся на классы:
    -> попадание в цель res=1
    -> столкновение res=0
    -> преследование res=-1"""
    k_p_list = np.exp(np.linspace(np.log(k_p_min), np.log(k_p_max), N))  # Logarithmic scale
    k_av_list = np.exp(np.linspace(np.log(k_a_min), np.log(k_a_max), N))  # Logarithmic scale
    start_time = datetime.now()
    tmp_count = 0
    for lvl in [2]:
        filename = 'storage/pid_const5_avoiding_' + name + '_' + str(lvl) + '.txt'
        # f = open(filename, 'a')
        f = open(filename, 'w')

        for k_p in k_p_list:
            for k_a in k_av_list:
                res = -1
                tmp_count += 1
                id_app = 0
                print(Fore.CYAN + f'Подбор ПД-к-в: {tmp_count}/{N**2}; время={datetime.now() - start_time}'
                    + Style.RESET_ALL)
                o = AllProblemObjects(if_PID_control=True, if_avoiding=True,
                                      dt=dt, k_p=k_p, k_av=k_a,
                                      T_max=T_max, level_avoid=lvl,
                                      if_talk=False, if_any_print=False,
                                      choice='5')

                for _ in range(int(T_max // dt)):
                    # Repulsion
                    if o.a.flag_fly[id_app] == 0:
                        _ = repulsion(o, id_app, u_a_priori=np.array([-random.uniform(0.001, 0.003), 0., 0.]))

                    # Control
                    o.time_step()
                    o.control_step(0)

                    # Docking
                    if (o.t - o.t_start[id_app]) > 100:
                        res = 0 if call_crash(o, o.a.r[0], o.r_ub, o.S, o.taken_beams) else res
                        if not res:
                            break
                    if o.get_discrepancy(id_app=0) < o.d_to_grab:
                        res = 1
                        break

                f.write(f'{k_p} {k_a} {lvl} {res}\n')
        # reader_avoid_field_params_search(name=name)
        f.close()

def reader_repulsion_error(name: str = ''):
    k_u_list = [0.01, 0.02, 0.05, 0.1, 0.2]
    k_w_list = [1e-5, 1e-4, 3e-4, 6e-4, 1e-3]
    filename = 'storage/repulsion_error_' + name + '.txt'
    f = open(filename, 'r')
    w, k, tol = ([], [], [])
    for line in f:
        lst = line.split()
        if len(lst) > 1:
            w += [k_w_list[int(lst[0])]]
            k += [int(lst[1])]
            tol += [float(lst[2])]

    tol_line = [[] for _ in range(len(k_u_list))]
    count = [0 for _ in range(len(k_u_list))]
    w_max = 0.
    i_max = 0

    '''for i in range(len(tol)):
        if w_max < w[i]:
            w_max = w[i]
            for k in range(len(k_u_list)):
                if count[k[i]] > 0:
                    tol_line[k][i_max - 2] /= count[k]
                tol_line[k].append(0.)
            count = [0 for _ in range(len(k_u_list))]
        else:
            tol_line[k_a[i]][len(k_p_list) - 1] += tol[i]
            count[k_a[i]] += 1'''

    plt.scatter(np.array(w) + (np.array(k) - 2) * 1e-5, tol, c='c')
    plt.show()
    
    '''for line in f:
        lst = line.split()
        k_p += [float(lst[0])]
        k_a += [int(lst[1])]
        tol += [float(lst[2])]
        col += [int(lst[3])]
    f.close()

    tol_line = [[] for _ in range(len(k_ac_list))]
    count = [0 for _ in range(len(k_ac_list))]
    k_p_max = 0.
    k_p_list = []
    for i in range(len(tol)):
        if k_p_max < k_p[i]:
            k_p_max = k_p[i]
            k_p_list += [k_p[i]]
            for k in range(len(k_ac_list)):
                if count[k_a[i]] > 0:
                    tol_line[k][len(k_p_list) - 2] /= count[k]
                tol_line[k].append(0.)
            count = [0 for _ in range(len(k_ac_list))]
        else:
            tol_line[k_a[i]][len(k_p_list) - 1] += tol[i]
            count[k_a[i]] += 1
    for k in range(len(k_ac_list)):
        tol_line[k][len(k_p_list) - 1] /= count[k]

    if lng == 'ru':
        title = "Подбор коэффициентов ПД-регулятора"
        x_label = "Коэффициент k_p"
        y_label = "Точность"
    else:
        title = "PD-controller coeffitient enumeration"
        x_label = "k_p coeffitient"
        y_label = "Tolerance"

    clr = [['aqua', 'violet', 'limegreen', 'gold', 'lightcoral'],
           ['steelblue', 'purple', 'green', 'goldenrod', 'firebrick']]
    plt.title(title)
    for i in range(len(k_ac_list)):
        plt.plot(k_p_list, tol_line[i], c=clr[0][i], label=k_ac_list[i])
        for kp in k_p_list:
            plt.plot([kp * (1 + eps * (i - 2))] * 2, [0., 20.], c=clr[0][i])
    for i in range(len(tol)):
        plt.scatter(k_p[i] * (1 + eps * (k_a[i] - 2)), tol[i], c=clr[col[i]][k_a[i]])
    plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()'''


def plot_repulsion_error(name: str = '', N: int = 40, dt: float = 1., T_max: float = 1000.):
    k_u_list = [0.01, 0.02, 0.05, 0.1, 0.2]
    k_w_list = [1e-5, 1e-4, 3e-4, 6e-4, 1e-3]
    start_time = datetime.now()
    filename = 'storage/repulsion_error_' + name + '.txt'
    f = open(filename, 'w')
    # errors = [[[] for _ in range(4)] for _ in range(len(k_u_list))]
    clr = ['plum', 'skyblue', 'darkgreen', 'maroon']
    tmp = 0
    id_app = 0
    for k in range(len(k_u_list)):
        for w in range(len(k_w_list)):
            u = None
            for j in range(N):
                tmp += 1
                print(Fore.CYAN + f"Невязка от погрешности отталкивания: {tmp}/{N*len(k_u_list)*len(k_w_list)}; "
                                f"время={datetime.now() - start_time}")
                f_min = 1e5
                o = AllProblemObjects(dt=dt, T_max=T_max, if_any_print=False, choice='4', method='shooting',
                                    d_crash=None, d_to_grab=None, if_talk=False, floor=7)
                for j in range(184):
                    o.s.flag[j] = np.array([1, 1])
                o.w = np.array([0., k_w_list[w], 0.])
                o.om_update()
                for i_time in range(int(T_max // dt)):
                    # Docking
                    f_min = min(f_min, o.get_discrepancy(id_app=0))

                    # Repulsion
                    if o.a.flag_fly[id_app] == 0:
                        u = repulsion(o, id_app) if u is None else \
                            repulsion(o, id_app, u_a_priori=velocity_spread(u, k_u_list[k]))

                    # Control
                    o.time_step()
                # errors[k][int(choice) - 1].append(f_min)
                f.write(f"{w} {k} {f_min}\n")
    f.close()

def crawling(u_crawl: float = 0.01):
    o = AllProblemObjects(choice='2', floor=10)
    total_length = 0.
    for i in range(o.s.n_beams):
        i_floor = int((o.s.r1[i][0] - o.s.r1[6][0]) // o.s.length[6]) + 1
        # print(f"i:{i} floor:{i_floor}")
        if i_floor:
            total_length += 2 * (o.s.length[0] + o.s.length[6] + o.s.container_length)
        else:
            total_length += 2 * (o.s.length[0] + o.s.container_length)
        print(f"{total_length / u_crawl} секунд: установлен стержень id:{i}")
    print(f"Время сборки: {total_length / u_crawl} секунд")

def reader_dv_col_noncol_difference(name: str = '', plot_name: str = ''):
    filename = 'storage/dv_col_noncol_difference_' + name + '.txt'
    f = open(filename, 'r')
    x = [[] for _ in range(6)]
    tmp, j, j_tmp = ([], 0, 0)
    for line in f:
        lst = line.split()
        if int(lst[2]) != j:
            j = int(lst[2])
            x[j_tmp].append(np.sum(tmp))
            if int(lst[2]) == 0:
                print(f'next: {np.sum(tmp)} / {len(tmp)}')
                j_tmp += 1
        tmp.append(float(lst[3]))
    x[j_tmp].append(np.sum(tmp))
    labels = []
    for cnstr in CONSTRUCTION_LIST:
        for d_crash in [None, 0.2]:
            labels.append(f"Конструкция {cnstr}\nСтолкновение {d_crash}")
    plt.title("Затраты характеристической скорости" + plot_name)
    plt.boxplot(x, labels=labels)  # [i], c=clrs[i])
    plt.show()
    f.close()

def dv_col_noncol_difference(name: str = '', dt: float = 0.1, t_max: float = 2000, u_max: float = 0.01, times: int = 5,
                             w_twist: float = 0.):
    """Функция показывает boxplot затрат характеристической скорости для разных типов конструкций с
    учётом столкновения и без"""
    from main_non_interface import iteration_func
    from datetime import datetime
    filename = 'storage/dv_col_noncol_difference_' + name + '.txt'
    f = open(filename, 'w')
    start_time = datetime.now()
    counter = 0
    for cnstr in CONSTRUCTION_LIST:
        for d_crash in [None, 0.2]:
            for j in range(times):
                counter += 1
                o = AllProblemObjects(w_twist=w_twist,
                                      e_max=1e5,
                                      j_max=1e5,
                                      R_max=1000.,
                                      method='diffevolve+shooting+pd',

                                      dt=dt, T_max=t_max, u_max=u_max,
                                      choice=cnstr, floor=7, d_crash=d_crash,
                                      N_apparatus=1, if_any_print=False,
                                      file_reset=False, if_avoiding=d_crash is not None)
                print(Fore.CYAN + f"Затраты характеричтической скорости: {counter}/{3*2*times} | "
                                  f"t:{datetime.now() - start_time}" + Style.RESET_ALL)
                tmp = random.randint(180, 400) if cnstr == '4' \
                    else (random.randint(20, 40) if cnstr == '2'
                          else random.randint(0, 20))
                for i in range(tmp):
                    o.s.flag[i] = np.array([1, 1])
                for _ in range(int(o.T_total // dt)):
                    iteration_func(o)
                    f.write(f"{cnstr} {d_crash} {j} {dt * np.linalg.norm(o.a_self)}\n")
                    if o.s.flag[tmp + times][0]:
                        break
    f.close()

def reader_dv_from_w_twist(name: str = '', plot_name: str = '', y_lim: int = 700):
    filename = 'storage/dv_from_w_twist_' + name + '.txt'
    f = open(filename, 'r')
    tmp, j, i_tmp = ([], 0, 0)
    w_twist = [0.00001, 0.0001, 0.0002, 0.0005]
    x = [[[] for _ in range(len(w_twist))] for _ in range(4)]  # x[0] = [[...], [...], [...]] len = w, x[0][0] = boxplot
    for line in f:
        lst = line.split()
        if int(lst[3]) != j:
            j = int(lst[3])
            i_tmp = (lst[2] != 'None') + 2 * (lst[0] != '0.1')
            x[i_tmp][w_twist.index(float(lst[1]))].append(np.sum(tmp))
            if int(lst[3]) == 0:
                print(f'next: {np.sum(tmp)}')
        tmp.append(float(lst[4]))
    x[i_tmp][w_twist.index(float(lst[1]))].append(np.sum(tmp))
    plt.title("Затраты характеристической скорости" + plot_name)

    labels = ["e=10%, d=None", "e=10%, d=0.2 м", "e=1%, d=None", "e=1%, d=0.2 м"]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.boxplot(x[i], labels=w_twist)
        plt.title(labels[i])
        plt.xlabel('Угловая скорость закрутки, рад/с', fontsize=FONTSIZE)
        plt.ylabel('Затраты ΔV, м/с', fontsize=FONTSIZE)
        plt.ylim([0, y_lim])

    plt.show()
    f.close()

def dv_from_w_twist(name: str = '', dt: float = 0.1, t_max: float = 2000, u_max: float = 0.01, times: int = 5):
    from main_non_interface import iteration_func
    from datetime import datetime
    filename = 'storage/dv_from_w_twist_' + name + '.txt'
    f = open(filename, 'w')
    start_time = datetime.now()
    counter = 0
    for e_max in [0.1, 0.01]:
        for w_twist in [0.00001, 0.0001, 0.0002, 0.0005]:
            for d_crash in [None, 0.2]:
                for j in range(times):
                    counter += 1
                    o = AllProblemObjects(w_twist=w_twist,
                                          e_max=e_max,
                                          j_max=1e5,
                                          R_max=1000.,
                                          method='diffevolve+shooting+pd',

                                          dt=dt, T_max=t_max, u_max=u_max,
                                          choice='4', floor=7, d_crash=d_crash,
                                          N_apparatus=1, if_any_print=False,
                                          file_reset=False)
                    print(Fore.CYAN + f"Затраты характеричтической скорости: {counter}/{2*4*2*times} | "
                                      f"t:{datetime.now() - start_time}" + Style.RESET_ALL)
                    tmp = random.randint(180, 250)
                    for i in range(tmp):
                        o.s.flag[i] = np.array([1, 1])
                    for _ in range(int(o.T_total // dt)):
                        iteration_func(o)
                        f.write(f"{e_max} {w_twist} {d_crash} {j} {dt * np.linalg.norm(o.a_self)}\n")
                        if o.s.flag[tmp + 10][0]:
                            break
    f.close()

def reader_full_bundle_of_trajectories(name: str = '', n_p: int = 10, n_t: int = 10):
    o = AllProblemObjects(choice='3')
    o.coordinate_system = 'body'
    filename0 = 'storage/full_bundle_param_' + name + '.txt'
    filename1 = 'storage/full_bundle_lines_' + name + '.txt'
    f0 = open(filename0, 'r')
    f1 = open(filename1, 'r')
    phi_list = np.linspace(-np.pi, np.pi, n_p, endpoint=False)
    theta_list = np.linspace(-np.pi / 2, np.pi / 2, n_t, endpoint=False)
    x, y = np.meshgrid(phi_list, theta_list)
    z = x + y  # z[y][x] z[theta][phi]
    dr, e, u, phi, theta = ([], [], [], [], [])
    for line in f0:
        lst = line.split()
        dr += [float(lst[0])]
        e += [float(lst[1])]
        u += [float(lst[2])]
        phi += [float(lst[3])]
        theta += [float(lst[4])]
        z[np.where(theta_list == float(lst[4]))[0][0]][np.where(phi_list == float(lst[3]))[0][0]] = float(lst[0])
        # z[theta_list.index(float(lst[4]))][phi_list.index(float(lst[3]))] = float(lst[0])
    lines = [[] for _ in range(len(e))]
    tmp = 0
    for line in f1:
        lst = line.split()
        for i in range(len(lst)):
            lines[tmp] += [float(lst[i])]
        tmp += 1

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)
    plt.show()
    msh = plot_iterations_new(o).color("silver")
    for i in range(len(lines)):
        msh += fig_plot(o, lines[i])
    show(msh, __doc__, viewup="xz", axes=0, bg='white', zoom=1, size=(1920, 1080)).close()
    f0.close()
    f1.close()


def full_bundle_of_trajectories(name: str = '', dt: float = 0.1, t_max: float = 5000, u0: float = 0.01, n_p: int = 10,
                                n_t: int = 10, control: any = None):
    """Разброс траекторий вокруг для качественной оценки
    По совместительству демонстрация несанкционированного хлопка в Барселоне"""
    filename0 = 'storage/full_bundle_param_' + name + '.txt'
    filename1 = 'storage/full_bundle_lines_' + name + '.txt'
    f0 = open(filename0, 'w')
    f1 = open(filename1, 'w')
    phi_list = np.linspace(-np.pi, np.pi, n_p, endpoint=False)
    theta_list = np.linspace(-np.pi / 2, np.pi / 2, n_t, endpoint=False)
    u_list = [u0]  # задел на будущее
    o = AllProblemObjects(dt=dt, T_max=t_max, choice='3', u_max=u0, if_testing_mode=True, a_pid_max=1e5)
    o.repulse_app_config(id_app=0)
    start_time = datetime.now()
    tmp = 0
    for u in u_list:
        for phi in phi_list:
            for theta in theta_list:
                tmp += 1
                o.my_print(f"Разброс траекторий {tmp}/{len(u_list)*len(phi_list)*len(theta_list)} | "
                           f"t:{datetime.now() - start_time}")
                dr, _, e, V, R, j, _, line = calculation_motion(o=o, u=polar2dec(u, phi, theta), T_max=t_max, id_app=0,
                                                                interaction=True, line_return=True, control=control)
                f0.write(f"{np.linalg.norm(dr)} {e} {u} {phi} {theta} {dt} {t_max}\n")
                for l in line:
                    f1.write(f"{l} ")
                f1.write(f"\n")
    f0.close()
    f1.close()

def full_bundle_of_trajectories_controlled(name: str = '', dt: float = 0.5, t_max: float = 5000, n_p: int = 10,
                                           n_t: int = 10, control: float = 1e-5):
    """Разброс траекторий вокруг для качественной оценки
    По совместительству демонстрация несанкционированного хлопка в Барселоне"""
    filename = 'storage/full_bundle_controlled__' + name + '.txt'
    phi_list = np.linspace(-np.pi, np.pi, n_p, endpoint=False)
    theta_list = np.linspace(-np.pi / 2, np.pi / 2, n_t, endpoint=False)
    start_time = datetime.now()
    tmp = 0
    lines = [[] for _ in range(n_p * n_t)]
    Radius_orbit = 6800e3
    mu = 5.972e24 * 6.67408e-11
    w_hkw = np.sqrt(mu / Radius_orbit ** 3)

    def rv_right_part(rv, a):
        return np.array([rv[3], rv[4], rv[5], a[0], a[1], a[2]])

    def rk4_acceleration(r, v, a):
        rv = np.append(r, v)
        k1 = rv_right_part(rv, a)
        k2 = rv_right_part(rv + k1 * dt / 2, a)
        k3 = rv_right_part(rv + k2 * dt / 2, a)
        k4 = rv_right_part(rv + k3 * dt, a)
        rv = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return rv[0:3] + r, rv[3:6] + v

    def get_hkw_acceleration(rv):
        return np.array([-2 * w_hkw * rv[5],
                         -w_hkw ** 2 * rv[1],
                         2 * w_hkw * rv[3] + 3 * w_hkw ** 2 * rv[2]])

    for phi in phi_list:
        for theta in theta_list:
            r = np.zeros(3)
            v = np.zeros(3)
            a = polar2dec(control, phi, theta)
            lines[tmp] += [r[0], r[1], r[2]]
            for _ in range(int(t_max // dt)):
                r, v = rk4_acceleration(r, v, a + get_hkw_acceleration(np.append(r, v)))
                lines[tmp] += [r[0], r[1], r[2]]
            tmp += 1
            print(f"Разброс управляемых траекторий {tmp}/{n_p * n_t} | "
                  f"t:{datetime.now() - start_time} | r={np.linalg.norm(r)}")

    '''f = open(filename, 'w')
    for l in lines:
        f.write(f"{l}\n")
    f.close()
    print("Записано в файл")'''

    o = AllProblemObjects()
    msh = []
    for i in range(len(lines)):
        msh += [fig_plot(o, lines[i])]
    show(msh, __doc__, viewup="xz", axes=0, bg='white', zoom=1, size=(1920, 1080)).close()


def reader_heatmap_function(name: str = '',  n_x: int = 10, n_y: int = 10, max_value: float = 1e5):
    import seaborn as sns
    filename = 'storage/heatmap_function_' + name + '.txt'
    f = open(filename, 'r')
    anw = [[0. for _ in range(n_x)] for _ in range(n_y)]
    for line in f:
        lst = line.split()
        if len(lst) == 2:
            n_x = int(lst[0])
            n_y = int(lst[1])
            anw = [[0. for _ in range(n_y)] for _ in range(n_x)]
        else:
            x = int(lst[0])
            y = int(lst[1])
            a = float(lst[2])
            c = int(lst[3])
            anw[x][y] = 1e9999 if c else min(a, max_value)
    f.close()

    rate = 20
    y_list = np.linspace(-15, 5, n_x)
    x_list = np.linspace(-10, 10, n_y)

    xlabels = ['{:4.2f}'.format(x) for x in x_list]
    ylabels = ['{:4.2f}'.format(y) for y in y_list]

    fig, axes = plt.subplots(1, 1)  # , figsize=(8, 6)
    ax = sns.heatmap(anw, ax=axes, cmap="plasma", xticklabels=xlabels, yticklabels=ylabels,
                     cbar_kws={'label': 'Невязка Δr, м'}, )
    ax.figure.axes[-1].yaxis.label.set_size(17)
    ax.set_xticks(ax.get_xticks()[::rate])
    ax.set_xticklabels(xlabels[::rate])
    ax.set_yticks(ax.get_yticks()[::rate])
    ax.set_yticklabels(ylabels[::rate])

    # plt.imshow(anw, cmap='plasma')
    axes.set_xlabel(f"x, м", fontsize=17)
    axes.set_ylabel(f"z, м", fontsize=17)
    # axes.colorbar()
    plt.show()


def heatmap_function(name: str = '', n_x: int = 10, n_y: int = 10, target_toward: bool = True, scipy_meh=False):
    """Цветная карта целевой функции"""
    from snsmylibs.numerical_methods import capturing_penalty
    filename = 'storage/heatmap_function_' + name + '.txt'
    f = open(filename, 'w')
    x_list = np.linspace(-15, 5, n_x)
    y_list = np.linspace(-10, 10, n_y)
    o = AllProblemObjects(choice='3', N_apparatus=1)
    temp = o.a.target[0].copy()
    # u = repulsion(o, 0, u_a_priori=np.zeros(3))
    o.repulse_app_config(id_app=0)
    if not target_toward:
        o.a.target[0] = temp
    o.t_reaction_counter = -1
    tmp = 0
    f.write(f"{n_x} {n_y}\n")
    for ix in range(n_x):
        for iy in range(n_y):
            tmp += 1
            o.flag_vision[0] = False
            r_sat = o.b_o(np.array([x_list[ix], y_list[iy], 0.]))
            if scipy_meh:
                anw = call_crash(o, r_sat, o.r_ub, o.S, iFunc=True, brf=True)
                f.write(f"{ix} {iy} {anw} {0}\n")
            else:
                o.a.r[0] = r_sat
                dr = o.S @ o.get_discrepancy(id_app=0, vector=True)
                n_crashes = call_crash(o, o.a.r[0], o.r_ub, o.S)
                visible, crhper = control_condition(o, 0, return_percentage=True)
                anw = capturing_penalty(o, dr, 0, 0, 0, 0, 0, n_crashes, visible, crhper, o.mu_ipm)
                o.my_print(f"Карта целевой функции: {tmp}/{n_x * n_y} : {visible} {crhper * 100}%", mode='c')
                f.write(f"{ix} {iy} {np.linalg.norm(anw)} {int(n_crashes)}\n")
    f.close()

def plot_iter_downgrade(filename: str = ''):
    """Функция смотрит че там сделала пристрелка и насколько быстро она работает и хоб шух в итерациях"""
    f = open('storage/iteration_docking' + filename + '.txt', 'r')
    tmp = []
    max_len = 0
    for line in f:
        lst = line.split()
        if int(lst[0]) == 0:
            color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            if len(tmp) > 1 and tmp[len(tmp) - 1] > 0.5:
                plt.plot(np.arange(len(tmp)), tmp, c=color)
                plt.scatter(np.arange(len(tmp)), tmp, c=color)
            tmp = []
        tmp += [float(lst[1])]
        max_len = max(max_len, len(tmp))
    plt.plot(np.arange(len(tmp)), tmp, c=color)
    plt.scatter(np.arange(len(tmp)), tmp, c=color)
    plt.plot([0, max_len - 1], [0.5, 0.5], c='b', label='зона захвата')
    plt.plot([0, max_len - 1], [0, 0], c='b')
    plt.xlabel('Количество шагов, безразм.', fontsize=13)
    plt.ylabel('Невязка, м', fontsize=13)
    plt.legend()
    f.close()
    plt.show()
