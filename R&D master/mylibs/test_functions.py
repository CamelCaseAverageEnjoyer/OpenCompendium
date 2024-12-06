import matplotlib.pyplot as plt

from all_objects import *
from vedo import *


def def_o():
    o1 = AllProblemObjects(if_talk=False)
    o1.w = np.array([1e-3, -3e-5, 0])
    o1.om_update()
    return o1

def test_time_of_calculating(u: any, dt: float = 1., T: float = 1000., n: int = 1):
    from datetime import datetime
    o = AllProblemObjects(if_talk=False, dt=dt)
    o.repulse_app_config(id_app=0)
    for i in range(n):
        time_start = datetime.now()
        _, _, _, _, _, _, _, _, _, _ = calculation_motion(o=o, u=u, T_max=T, id_app=0, interaction=True, check_visible=False)
        print(f"Время выполнения {i+1}/{n}: {datetime.now() - time_start}")

def test_stitching_collision_function(n: int = 10):
    """Функция показывает heatmap для 1 стержня чтобы показать что функция гладенькая"""
    r1 = np.array([-1., 0., 0.])
    r2 = np.array([1., 0., 0.])
    diam = 0.5
    anw = [[0. for _ in range(n)] for _ in range(n)]
    x_list = np.linspace(-2, 2, n)
    y_list = np.linspace(-2, 2, n)

    for i in range(n):
        for j in range(n):
            x = x_list[i]
            y = y_list[j]
            anw[i][j] = call_crash_internal_func(np.array([x, y, 0]), r1, r2, diam, return_dist=True)
    plt.imshow(anw, cmap='plasma')
    plt.colorbar()
    color_box = 'aqua'
    plt.plot([0.375*n - 0.5, 0.375*n - 0.5], [0.25*n - 0.5, 0.75*n - 0.5], c=color_box)
    plt.plot([0.625*n - 0.5, 0.625*n - 0.5], [0.25*n - 0.5, 0.75*n - 0.5], c=color_box)
    plt.plot([0.375*n - 0.5, 0.625*n - 0.5], [0.25*n - 0.5, 0.25*n - 0.5], c=color_box)
    plt.plot([0.375*n - 0.5, 0.625*n - 0.5], [0.75*n - 0.5, 0.75*n - 0.5], c=color_box)
    plt.show()

def test_center_mass(u_max=None, dt=1., T_max=1000.):
    o = AllProblemObjects(if_talk=False, u_max=u_max, dt=dt)
    u = repulsion(o, 0, u_a_priori=np.array([random.uniform(-o.u_max/3, u_max/3) for _ in range(3)]))
    m_extra, M_without = o.get_masses(0)
    print(f"m_extra {m_extra}")
    print(f"M_without {M_without}")
    rx = []
    ry = []
    rz = []
    Rx = []
    Ry = []
    Rz = []
    rcx = []
    rcy = []
    rcz = []
    t = []
    for tt in range(int(T_max//dt)):
        o.time_step()
        t += [tt * dt]
        rx += [o.a.r[0][0]]
        ry += [o.a.r[0][1]]
        rz += [o.a.r[0][2]]
        Rx += [o.r_ub[0]]
        Ry += [o.r_ub[1]]
        Rz += [o.r_ub[2]]
        # print(f"Есть {M_without/m_extra}, надо {o.a.r[0][0]/o.R[0]} {o.a.r[0][1]/o.R[1]} {o.a.r[0][2]/o.R[2]}")
        tmp = (M_without * o.r_ub + m_extra * o.a.r[0]) / (m_extra + M_without)
        rcx += [tmp[0]]
        rcy += [tmp[1]]
        rcz += [tmp[2]]
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Проверка центра масс ОСК", fontsize=15)
    ax0.set_title('Компонента x')
    ax0.plot(t, rx, c='c', label='апп')
    ax0.plot(t, Rx, c='m', label='конст')
    ax0.plot(t, rcx, c='k', label='о.ц.м.')
    ax1.set_title('Компонента y')
    ax1.plot(t, ry, c='c', label='апп')
    ax1.plot(t, Ry, c='m', label='конст')
    ax1.plot(t, rcy, c='k', label='о.ц.м.')
    ax2.set_title('Компонента z')
    ax2.plot(t, rz, c='c', label='апп')
    ax2.plot(t, Rz, c='m', label='конст')
    ax2.plot(t, rcz, c='k', label='о.ц.м.')
    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax0.set_xlabel('время t, с')
    ax1.set_xlabel('время t, с')
    ax2.set_xlabel('время t, с')
    ax0.set_ylabel('x, м')
    ax1.set_ylabel('y, м')
    ax2.set_ylabel('z, м')
    plt.show()

def test_full_energy(order, w=0.001, dt=1., T_max=1000., show_rate: int = 10):
    # Сохранение полной энергии
    result = True
    e = 10**(-order)
    clrs = [['#CDC673', '#CD6090', '#ADFF2F'], ['#8B864E', '#8B3A62', '#00CD00'], ['#008000', '#4B0082', '#008000']]
    styles = ['-', ':', '--']
    tmp = -1
    for begin_rotation in ['xx']:  # , 'xy', 'xz']:
        tmp += 1
        o = AllProblemObjects(if_talk=False, dt=dt, choice='2', floor=10, choice_complete=True, N_apparatus=0, begin_rotation=begin_rotation)
        o.w = np.array([0., w, 0.])
        o.om_update()
        T0 = o.get_kinetic_energy()
        U0 = o.get_potential_energy()
        T = [o.get_kinetic_energy()]
        U = [o.get_potential_energy() - U0]
        E_0 = o.get_kinetic_energy() + o.get_potential_energy() - U0
        E_list = [E_0]
        t = [0.]
        for i in range(int(T_max/dt)):
            o.time_step()
            if i % show_rate == 0:
                T.append(o.get_kinetic_energy())
                U.append(o.get_potential_energy() - U0)
                E = o.get_kinetic_energy() + o.get_potential_energy() - U0
                E_list.append(E)
                t.append(o.dt * i)
                E_max = max(abs(E), max(abs(T[len(T) - 1]), abs(U[len(T) - 1])))
                if E_max > 0 and abs(E - E_0) / E_max > e:
                    result = False
                if E_max > 0 and abs(E - E_0) / E_max > e:
                    result = False

        T1 = 2*np.pi/o.w_hkw
        for i in range(int(np.floor(T_max / T1))):
            plt.plot([T1 * (i + 1), T1 * (i + 1)], [np.min(U), np.max(T)], c='#5F9EA0')
        
        plt.plot(t, U, c=clrs[tmp][0], label=f'потенциальная {begin_rotation}', linestyle=styles[tmp])
        plt.plot(t, T, c=clrs[tmp][1], label=f'вращательная {begin_rotation}', linestyle=styles[tmp])
        plt.plot(t, E_list, c=clrs[tmp][2], label=f'полная {begin_rotation}', linestyle=styles[tmp])
    plt.title("График энергий")
    plt.xlabel("время, c")
    plt.ylabel("энергия, Дж")
    plt.legend()
    draw_reference_frames(o, size=10, showing=True)
    return result


def test_rotation(order, o=None, dt=1., w=0.001, T_max=1000.):
    # Правильность задания матриц вращения
    result = True
    eps = 10**(-order)
    o = def_o() if o is None else o
    o.w = np.array([0, -w, 0])
    o.om_update()
    for i in range(int(T_max/dt)):
        o.time_step()
        if i % 10 == 0:
            for j in range(3):
                if o.U.T[0][j] > eps:
                    if (o.U.T[0][j] - np.linalg.inv(o.U)[0][j]) / o.U.T[0][j] > eps:
                        result = False
                if o.A.T[0][j] > eps:
                    if (o.A.T[0][j] - np.linalg.inv(o.A)[0][j]) / o.A.T[0][j] > eps:
                        result = False
                if np.linalg.inv(o.U)[0][j] > eps:
                    if (o.U.T[0][j] - np.linalg.inv(o.U)[0][j]) / np.linalg.inv(o.U)[0][j] > eps:
                        result = False
                if np.linalg.inv(o.S)[0][j] > eps:
                    if (o.S.T[0][j] - np.linalg.inv(o.S)[0][j]) / np.linalg.inv(o.S)[0][j] > eps:
                        result = False
    a = [0, 1, 2]
    c = np.array([0., 1., 2.])
    A = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
    for _ in range(20):
        Lu = np.random.rand(4)
        Lu /= np.linalg.norm(Lu)
        Ls = np.random.rand(4)
        Ls /= np.linalg.norm(Ls)
        U = quart2dcm(np.double(Lu))
        S = quart2dcm(np.double(Ls))
        if abs(1. - np.linalg.det(U)) > eps:
            result = False
        if abs(1. - np.linalg.det(S)) > eps:
            result = False
        b = o.i_o(o.o_i(o.i_o(o.o_i(a, U=U), U=U), U=U), U=U)
        for _ in range(50):
            b = o.i_o(o.o_i(o.i_o(o.o_i(b, U=U), U=U), U=U), U=U)
        for i in range(3):
            if abs(1. - np.linalg.norm([U[0][i], U[1][i], U[2][i]])) > eps:
                result = False
            if abs(1. - np.linalg.norm([S[0][i], S[1][i], S[2][i]])) > eps:
                result = False
            for j in range(3):
                if abs(U.T[j][i] - np.linalg.inv(U)[j][i]) > eps:
                    result = False
                if abs(A[j][i] - o.b_i(o.i_b(A))[j][i]) > eps:
                    result = False
            if abs(c[i] - o.i_o(o.o_i(c, U=U), U=U)[i]) > eps:
                result = False
            if abs(a[i] - o.i_o(o.o_i(b, U=U), U=U)[i]) > eps:
                result = False
            if abs(c[i] - o.o_b(o.b_o(c, S=S), S=S)[i]) > eps:
                result = False
            if abs(a[i] - o.o_b(o.b_o(a, S=S), S=S)[i]) > eps:
                result = False
            if abs(c[i] - o.i_b(o.b_i(c))[i]) > eps:
                result = False
            if abs(a[i] - o.i_b(o.b_i(a))[i]) > eps:
                result = False
            if abs(o.o_b(o.i_o(c))[i] - o.i_b(c)[i]) > eps:
                result = False
            if abs(o.b_o(o.i_b(c))[i] - o.i_o(c)[i]) > eps:
                result = False
    return result


def test_runge_kutta(order, o=None, dt=1., w=0.001, T_max=1000.):
    result = True
    eps = 10 ** (-order)
    o = def_o() if o is None else o
    o.w = np.array([0, -w, 0])
    o.om_update()
    r = np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)])
    v = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
    c = get_c_hkw(r, v, o.w_hkw)
    for i in range(int(T_max/dt)):
        r, v = o.rk4_acceleration(r, v, [0, 0, 0])
    r_h = r_hkw(c, o.w_hkw, T_max)
    v_h = v_hkw(c, o.w_hkw, T_max)
    for i in range(3):
        if abs(r[i] / r_h[i]) < eps:
            result = False
        if abs(r_h[i] / r[i]) < eps:
            result = False
        if abs(v[i] / v_h[i]) < eps:
            result = False
        if abs(v_h[i] / v[i]) < eps:
            result = False
    return result

def test_collision_map(n: int = 10, x_boards: list = np.array([-10, 10]), z_boards: list = np.array([-5, 10])):
    o = AllProblemObjects(choice='0', N_apparatus=0)
    x_list = np.linspace(x_boards[0], x_boards[1], n)
    z_list = np.linspace(z_boards[0], z_boards[1], n)
    points = None
    for x in x_list:
        for z in z_list:
            clr = 'm' if call_crash(o, np.array([x, 0., z]), o.r_ub, o.S) else 'c'
            if points is None:
                points = [vedo.Point(np.array([x, 0., z]), c=clr)]
            else:
                points += [vedo.Point(np.array([x, 0., z]), c=clr)]

    points.append(plot_iterations_new(o).color("silver", alpha=0.4))
    show(points, __doc__, viewup="z", axes=1, bg='white', zoom=1, size=(1920, 1080)).close()
