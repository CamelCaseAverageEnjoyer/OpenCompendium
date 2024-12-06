# 3d plot libraries
from vedo import dataurl, Mesh, Plotter
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import stl

# Standard libraries
import matplotlib.pyplot as plt
from scipy import spatial
from PIL import Image
from stl import mesh
import vedo

# Local libraries
from mylibs.tiny_functions import *

RADIUS_PER_LEN = 0.1
N_BEAM_ROUND = 50
LINE_WIDTH = 1


def color_between(tau: float):
    c0 = (0.8, 0.8, 0.8)
    c1 = (1.0, 0.0, 1.0)
    if tau <= 0:
        return c1
    if tau >= 1:
        return c0
    return 0.8 * tau + 1.0 * (1 - tau), 0.8 * tau, 0.8 * tau + 1.0 * (1 - tau)


def line_chaos(x_lim: float = 20., y_lim: float = 20., z_lim: float = 20.):
    """Функция создания линии помех на vedo-изображении для видимости нарушения жёстких ограничений. \n
    Используется вместо вылета программы."""
    line = [-x_lim, -y_lim, -z_lim,
            x_lim, y_lim, z_lim,
            -x_lim, -y_lim, z_lim,
            x_lim, y_lim, -z_lim,
            -x_lim, y_lim, -z_lim,
            x_lim, -y_lim, z_lim,
            x_lim, -y_lim, -z_lim,
            -x_lim, y_lim, z_lim]
    return line


def line_target(r, d: float = 0.5):
    """Рисует кружочек вокруг цели."""
    line = []
    N = 40
    for i in range(N):
        line = np.append(line, np.array(r)+d*np.array([np.sin(i/N*2*np.pi), 0., np.cos(i/N*2*np.pi)]))
    for i in range(N):
        line = np.append(line, np.array(r)+d*np.array([0., np.sin(i/N*2*np.pi), np.cos(i/N*2*np.pi)]))
    for i in range(round(N / 10)):
        line = np.append(line, np.array(r)+d*np.array([0., np.sin(i/N*2*np.pi), np.cos(i/N*2*np.pi)]))
    for i in range(N):
        line = np.append(line, np.array(r)+d*np.array([np.sin(i/N*2*np.pi), np.cos(i/N*2*np.pi), 0.]))
    return line


def plot_by_y(y, x=None, name=None, color: str = 'slategray', color_zero: str = 'powderblue'):
    """Функция, показывающая график; \n
    Также показываются название графика, min/max значения функции."""
    fig, ax = plt.subplots()
    if x:
        ax.plot(x, y, c=color)
        ax.plot(x, np.zeros(len(x)), c=color_zero)
    else:
        ax.plot(range(len(y)), y, c=color)
        ax.plot(range(len(y)), np.zeros(len(y)), c=color_zero)
    ax.text(len(y) * 2 / 3, np.max(y) * 17.9 / 20 + np.min(y) * 2 / 20, 'min: ' + str(np.min(y)),
            rotation=0,
            fontsize=8 * 2)
    ax.text(len(y) * 2 / 3, np.max(y) * 17 / 20 + np.min(y) * 3 / 20, 'max:' + str(np.max(y)),
            rotation=0,
            fontsize=8 * 2)
    if name:
        ax.text(len(y) * 6.7 / 11, np.max(y) * 19 / 20 + np.min(y) / 20, str(name),
                fontsize=16 * 2)
    fig.set_figwidth(6 * 2)
    fig.set_figheight(4 * 2)
    plt.show()
    

def fig_plot(o, line_0, point=None, color=None):
    """Функция распаковки линии, построения vedo.Line"""
    global LINE_WIDTH
    N = int(np.floor(len(line_0) / 3))
    x = [line_0[3 * i + 0] for i in range(N)]
    y = [line_0[3 * i + 1] for i in range(N)]
    z = [line_0[3 * i + 2] for i in range(N)]
    if o.coordinate_system == 'real':
        vertices = [o.U.T @ np.array([x[i], y[i], z[i]]) for i in range(N)]
    else:
        vertices = [[x[i], y[i], z[i]] for i in range(N)]
    color = (238/255, 130/255, 238/255) if color is None else color
    line = vedo.Line(vertices, c=color, lw=LINE_WIDTH)
    if point is not None: 
        line += vedo.Point(point, c='c')
    return line


def draw_vector(ax, v, r0=None, v0=None, clr: str = 'm', style: str = '-'):
    """Функия рисует стрелочку в осях matplotlib.pyplot"""
    r0 = np.array([0., 0., 0.]) if r0 is None else r0
    rate = 0.9
    width = 0.02
    
    if v[2] > v[1] and v[2] > v[0]:
        b = my_cross(v, [0, 1, 0])
    else:
        if v[0] > v[1] and v[0] > v[2]:
            b = my_cross(v, [0, 0, 1])
        else:
            b = my_cross(v, [1, 0, 0])
    if np.linalg.norm(b) > 0:
        b *= width*np.linalg.norm(v)/np.linalg.norm(b)
    if v0 is None:
        ax.plot([r0[0], r0[0]+v[0]], [r0[1], r0[1]+v[1]], [r0[2], r0[2]+v[2]], c=clr, ls=style)
        ax.plot([r0[0]+v[0]*rate+b[0], r0[0]+v[0]], [r0[1]+v[1]*rate+b[1], r0[1]+v[1]], [r0[2]+v[2]*rate+b[2],
                                                                                         r0[2]+v[2]], c=clr)
        ax.plot([r0[0]+v[0]*rate-b[0], r0[0]+v[0]], [r0[1]+v[1]*rate-b[1], r0[1]+v[1]], [r0[2]+v[2]*rate-b[2],
                                                                                         r0[2]+v[2]], c=clr)
    else:
        ax.plot([v0[0], v[0]], [v0[1], v[1]], [v0[2], v[2]], c=clr, ls=style)
        ax.plot([v0[0]+(v[0]-v0[0])*rate+b[0], v[0]], [v0[1]+(v[1]-v0[1])*rate+b[1], v[1]],
                [v0[2]+(v[2]-v0[2])*rate+b[2], v[2]], c=clr)
        ax.plot([v0[0]+(v[0]-v0[0])*rate-b[0], v[0]], [v0[1]+(v[1]-v0[1])*rate-b[1], v[1]],
                [v0[2]+(v[2]-v0[2])*rate-b[2], v[2]], c=clr)
        
        
def draw_reference_frames(o, size: int = 10, showing: bool = False):
    """Функция рисует системы координат"""
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(projection='3d')

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = o.Radius_orbit/2 * np.outer(np.cos(u), np.sin(v))
    y = o.Radius_orbit/2 * np.outer(np.sin(u), np.sin(v))
    z = o.Radius_orbit/2 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, alpha=0.3)
    draw_vector(ax=ax, v=[0., 0., o.Radius_orbit], clr='k', style='-.')
    draw_vector(ax=ax, v=[0., o.Radius_orbit, 0.], clr='k', style=':')
    draw_vector(ax=ax, v=[o.Radius_orbit, 0., 0.], clr='k')
    draw_vector(ax=ax, v=o.o_i(np.zeros(3)), clr='silver')
    draw_vector(ax=ax, v0=o.o_i(np.zeros(3)), v=o.o_i([1e6, 0, 0]), clr='darkslateblue')
    draw_vector(ax=ax, v0=o.o_i(np.zeros(3)), v=o.o_i([0, 1e6, 0]), clr='darkslateblue', style=':')
    draw_vector(ax=ax, v0=o.o_i(np.zeros(3)), v=o.o_i([0, 0, 1e6]), clr='darkslateblue', style='-.')
    draw_vector(ax=ax, v0=o.b_i(np.zeros(3)), v=o.b_i([1e6, 0, 0]), clr='aqua')
    draw_vector(ax=ax, v0=o.b_i(np.zeros(3)), v=o.b_i([0, 1e6, 0]), clr='aqua', style=':')
    draw_vector(ax=ax, v0=o.b_i(np.zeros(3)), v=o.b_i([0, 0, 1e6]), clr='aqua', style='-.')
    ax.axes.set_xlim3d(left=-o.Radius_orbit*1.2, right=o.Radius_orbit*1.2)
    ax.axes.set_ylim3d(bottom=-o.Radius_orbit*1.2, top=o.Radius_orbit*1.2)
    ax.axes.set_zlim3d(bottom=-o.Radius_orbit*1.2, top=o.Radius_orbit*1.2)
    if showing:
        plt.show()
    else:
        plt.savefig('storage/tmp_pic.png', transparent=False)
        plt.close()


def show_beam(r1, r2, flag_non_relative=0, raduis=None):
    """Возвращает точки основания цилиндра стержня"""
    global N_BEAM_ROUND, RADIUS_PER_LEN
    raduis = RADIUS_PER_LEN if raduis is None else raduis
    L = np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2 + (r1[2] - r2[2]) ** 2)
    if flag_non_relative > 0.5:
        radius = raduis
    else:
        radius = raduis * L
    x_up = [radius * np.cos(i / N_BEAM_ROUND * 2 * np.pi) for i in range(N_BEAM_ROUND)]
    y_up = [radius * np.sin(i / N_BEAM_ROUND * 2 * np.pi) for i in range(N_BEAM_ROUND)]
    z_up = [L for _ in range(N_BEAM_ROUND)]
    z_down = [0 for _ in range(N_BEAM_ROUND)]
    vertices_0 = np.array([[0.0, 0.0, 0.0] for _ in range(2 * N_BEAM_ROUND)])
    for i in range(N_BEAM_ROUND):
        vertices_0[2 * i] = [x_up[i], y_up[i], z_up[i]]
        vertices_0[2 * i + 1] = [x_up[i], y_up[i], z_down[i]]

    r_tmp = np.array([r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]]) / L
    tau = np.cross([0.0, 0.0, 1.0], r_tmp)
    if np.linalg.norm(tau) > 0:
        tau = tau / np.linalg.norm(tau)
    phi = np.arccos(np.array([0.0, 0.0, 1.0]) @ r_tmp)
    quaternion_tmp = np.array(
        [np.cos(phi / 2), tau[0] * np.sin(phi / 2), tau[1] * np.sin(phi / 2), tau[2] * np.sin(phi / 2)])
    A = quart2dcm(quaternion_tmp)
    vertices = np.array([[0.0, 0.0, 0.0] for _ in range(2 * N_BEAM_ROUND)])
    for i in range(2 * N_BEAM_ROUND):
        vertices[i] = np.linalg.inv(A) @ vertices_0[i] + np.array(r1)
    if (r_tmp[2] < 0) and (phi == np.pi):
        for i in range(2 * N_BEAM_ROUND):
            vertices[i] = vertices_0[i] + np.array(r2)
    return vertices


def draw_apparatus(a, l_h, r, h_up, h_front, phi_right_1, theta_right_1, theta_right_2, phi_left_1, theta_left_1,
                   theta_left_2):
    """Возвращает mesh аппарата"""
    h_right = np.cross(h_front, h_up) / np.linalg.norm(h_up) / np.linalg.norm(h_front) * a / 2
    h_left = - h_right

    # Вычисления правой руки
    h_right_1 = -h_up / np.linalg.norm(h_up) * l_h * np.sin(theta_right_1) * np.cos(
        phi_right_1) + h_front / np.linalg.norm(h_front) * l_h * np.sin(theta_right_1) * np.sin(
        phi_right_1) + h_right / np.linalg.norm(h_right) * l_h * np.cos(theta_right_1)
    tau = np.cross(h_right, h_right_1)
    if abs(np.linalg.norm(tau)) > 1e-6:
        tau = tau / np.linalg.norm(tau)  # Огромные проблемы когда theta_right_1 = 0, поэтому он не будет равен нулю )))
    b = np.cross(tau, h_right_1)
    if abs(np.linalg.norm(b)) > 1e-6:
        b = b / np.linalg.norm(b)  # на всякий случай?
    h_right_2 = h_right_1 / np.linalg.norm(h_right_1) * l_h * np.cos(theta_right_2) + b / np.linalg.norm(
        b) * l_h * np.sin(theta_right_2)

    # Вычисления левой руки
    h_left_1 = -h_up / np.linalg.norm(h_up) * l_h * np.sin(theta_left_1) * np.cos(
        phi_left_1) + h_front / np.linalg.norm(h_front) * l_h * np.sin(theta_left_1) * np.sin(
        phi_left_1) + h_left / np.linalg.norm(h_left) * l_h * np.cos(theta_left_1)
    tau = np.cross(h_left, h_left_1)
    if abs(np.linalg.norm(tau)) > 1e-6:
        tau = tau / np.linalg.norm(tau)  # Огромные проблемы когда theta_right_1 = 0, поэтому он не будет равен нулю )))
    b = np.cross(tau, h_left_1)
    if abs(np.linalg.norm(b)) > 1e-6:
        b = b / np.linalg.norm(b)  # на всякий случай?
    h_left_2 = h_left_1 / np.linalg.norm(h_left_1) * l_h * np.cos(theta_left_2) + b / np.linalg.norm(b) * l_h * np.sin(
        theta_left_2)

    # Отображение
    r1 = r - h_up / np.linalg.norm(h_up) * a / 2  # Главное тельце
    r2 = r + h_up / np.linalg.norm(h_up) * a / 2
    vertices = show_beam(r1, r2, 1, a / 2)
    hull = spatial.ConvexHull(vertices)
    faces = hull.simplices
    myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            myramid_mesh.vectors[i][j] = vertices[f[j], :]
    main_body = myramid_mesh

    r1 = r + h_right  # Плечо правой руки
    r2 = r + h_right + h_right_1
    vertices = show_beam(r1, r2, 0)
    hull = spatial.ConvexHull(vertices)
    faces = hull.simplices
    myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            myramid_mesh.vectors[i][j] = vertices[f[j], :]
    twist_lock = myramid_mesh
    main_body = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))

    r1 = r + h_right + h_right_1  # Предплечье правой руки
    r2 = r + h_right + h_right_1 + h_right_2
    vertices = show_beam(r1, r2, 0)
    hull = spatial.ConvexHull(vertices)
    faces = hull.simplices
    myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            myramid_mesh.vectors[i][j] = vertices[f[j], :]
    twist_lock = myramid_mesh
    main_body = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))

    r1 = r + h_left  # Плечо левой руки
    r2 = r + h_left + h_left_1
    vertices = show_beam(r1, r2, 0)
    hull = spatial.ConvexHull(vertices)
    faces = hull.simplices
    myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            myramid_mesh.vectors[i][j] = vertices[f[j], :]
    twist_lock = myramid_mesh
    main_body = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))

    r1 = r + h_left + h_left_1  # Предплечье левой руки
    r2 = r + h_left + h_left_1 + h_left_2
    vertices = show_beam(r1, r2, 0)
    hull = spatial.ConvexHull(vertices)
    faces = hull.simplices
    myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            myramid_mesh.vectors[i][j] = vertices[f[j], :]
    twist_lock = myramid_mesh
    main_body = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))

    return main_body


def orientation_taken_rod(o, id_app):
    n = (o.o_b(o.a.r[id_app]) - o.r_center) / np.linalg.norm(o.o_b(o.a.r[id_app]) - o.r_center)
    if np.linalg.norm(my_cross(n, [1., 0., 0.])) > 1e-4:
        b = my_cross(n, [1., 0., 0.])
        b /= np.linalg.norm(b)
    else:
        b = [0., 1., 0.]
    tau = my_cross(n, b)
    tau /= np.linalg.norm(tau)
    return n, b, tau


def plot_iterations_new(o):
    """Возвращает mesh конструкции"""
    from mylibs.control_function import avoiding_force
    diam_cylinders_if_not = 0.001
    main_body = None
    
    '''for b in range(o.N_beams):
        if np.sum(o.s.flag[b]) == 0:
            r1 = o.s.r1[b]
            r2 = o.s.r2[b]
            if o.coordinate_system == 'orbital':
                r1 = o.R + o.S.T @ (o.s.r1[b] - o.r_center)
                r2 = o.R + o.S.T @ (o.s.r2[b] - o.r_center)
            if o.coordinate_system == 'support':
                r1 = o.S.T @ (o.s.r1[b])
                r2 = o.S.T @ (o.s.r2[b])
            vertices = show_beam(r1, r2, 1, diam_cylinders_if_not)

            hull = spatial.ConvexHull(vertices)
            faces = hull.simplices
            myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    myramid_mesh.vectors[i][j] = vertices[f[j], :]
            if main_body is None:
                main_body = myramid_mesh
            else:
                twist_lock = myramid_mesh
                main_body = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))'''

    for b in range(o.s.n_beams):
        # if b not in taken_beams:  # CДЕЛАТЬ ПО НОРМАЛЬНОМУ
        if np.sum(o.s.flag[b]) > 0:
            r1 = o.s.r1[b]
            r2 = o.s.r2[b]
        elif b not in o.taken_beams:
            r1 = o.s.r_st[b]
            r2 = o.s.r_st[b] - np.array([o.s.length[b] + 0.1, 0, 0])
        else:  # Нет учёта нескольких аппаратов: взятый стержень
            _, _, tau = orientation_taken_rod(o, id_app=0)
            r1 = o.o_b(o.a.r[0]) + tau * o.s.length[b] / 2  # np.array([o.s.length[b] / 2, 0, 0])
            r2 = o.o_b(o.a.r[0]) - tau * o.s.length[b] / 2  
        if o.coordinate_system == 'orbital':
            r1 = o.b_o(r1)
            r2 = o.b_o(r2)
        if o.coordinate_system == 'support':
            r1 = o.S.T @ r1
            r2 = o.S.T @ r2
        if o.coordinate_system == 'real':
            r1 = o.U.T @ o.b_o(r1)
            r2 = o.U.T @ o.b_o(r2)

        vertices = show_beam(r1, r2, 1)
        hull = spatial.ConvexHull(vertices)
        faces = hull.simplices
        myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                myramid_mesh.vectors[i][j] = vertices[f[j], :]
        if main_body is None:
            main_body = myramid_mesh
        else:
            twist_lock = myramid_mesh
            main_body = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))
            
    for b in range(o.c.n):
        r1 = o.c.r1[b]
        r2 = o.c.r2[b]
        if o.coordinate_system == 'orbital':
            r1 = o.b_o(r1)
            r2 = o.b_o(r2)
        if o.coordinate_system == 'support':
            r1 = o.S.T @ r1
            r2 = o.S.T @ r2
        if o.coordinate_system == 'real':
            r1 = o.U.T @ o.b_o(r1)
            r2 = o.U.T @ o.b_o(r2)
        vertices = show_beam(r1, r2, 1, o.c.diam[b])
        hull = spatial.ConvexHull(vertices)
        faces = hull.simplices
        myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                myramid_mesh.vectors[i][j] = vertices[f[j], :]
        twist_lock = myramid_mesh
        main_body = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))

    verts_temp, faces_temp = [], []
    for i in range(len(main_body.v0)):
        verts_temp.append(main_body.v0[i])
        verts_temp.append(main_body.v1[i])
        verts_temp.append(main_body.v2[i])
        faces_temp.append([i * 3, i * 3 + 1, i * 3 + 2])
    ready_mesh = vedo.Mesh([verts_temp, faces_temp]).clean()
    return ready_mesh


def plot_apps_new(o):
    """Возвращает mesh аппаратов"""
    ready_mesh = None
    for id_app in range(o.a.n):
        r_tmp = o.S @ (o.a.r[id_app] - o.r_ub) + o.r_center
        if o.coordinate_system == 'orbital':
            r_tmp = o.a.r[id_app]
        if o.coordinate_system == 'support':
            r_tmp = (o.a.r[id_app] - o.r_ub) + o.r_center
        if o.coordinate_system == 'real':
            r_tmp = o.U.T @ o.a.r[id_app]
        r_tmp += 0.5 * r_tmp / np.linalg.norm(r_tmp)
        n, b, tau = orientation_taken_rod(o, id_app=id_app)
        if o.coordinate_system == 'orbital':
            vec_up = o.S.T @ n
            vec_front = o.S.T @ b
        if o.coordinate_system == 'real':
            vec_up = o.A.T @ n
            vec_front = o.A.T @ b
        main_body = mesh.Mesh(draw_apparatus(0.2, 0.3, r_tmp, vec_up, vec_front, 0,
                                             30 * np.pi / 180, 70 * np.pi / 180, 0, 30 * np.pi / 180,
                                             70 * np.pi / 180).data)
        verts_temp, faces_temp = [], []
        for i in range(len(main_body.v0)):
            verts_temp.append(main_body.v0[i])
            verts_temp.append(main_body.v1[i])
            verts_temp.append(main_body.v2[i])
            faces_temp.append([i * 3, i * 3 + 1, i * 3 + 2])
        app_color = "navy" if o.a.flag_beam[id_app] is None else "m"
        if ready_mesh is None:
            ready_mesh = vedo.Mesh([verts_temp, faces_temp]).clean().color(app_color)
        else:
            ready_mesh += vedo.Mesh([verts_temp, faces_temp]).clean().color(app_color)
    return ready_mesh


def draw_flat_arrow(vec, o, id_app: int, clr: str = 'c'):
    from vedo import FlatArrow
    x, y, z = o.a.r[id_app]
    tmp = np.array([0.1*vec[1] + 0.1*vec[2], 0.1*vec[0] + 0.1*vec[2], 0.1*vec[0] + 0.1*vec[1]])
    l1 = [np.array([x, y, z]) - tmp, np.array([x, y, z]) - tmp + vec]
    l2 = [np.array([x, y, z]) + tmp, np.array([x, y, z]) + tmp + vec]
    return FlatArrow(l1, l2, tip_size=1, tip_width=1).c(color=clr, alpha=1)


def avoid_field_internal_func(o, avoid_vectors, x_boards: list, z_boards: list, nx: int, nz: int):
    from vedo import FlatArrow
    from mylibs.control_function import avoiding_force
    x_list = np.linspace(x_boards[0], x_boards[1], nx)
    z_list = np.linspace(z_boards[0], z_boards[1], nz)

    i = 0
    forces = [np.zeros(3) for i in range(nx * nz)]
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
                l1 = [np.array([x, 0, z]), np.array([x, 0, z]) + force]
                l2 = [np.array([x + 0.1 * force[2], 0, z + 0.1 * force[0]]),
                      np.array([x + 0.1 * force[2], 0, z + 0.1 * force[0]]) + force]
                farr = FlatArrow(l1, l2, tip_size=1, tip_width=1).c(color='m', alpha=0.3)
                if avoid_vectors is None:
                    avoid_vectors = farr
                else:
                    avoid_vectors += farr
    return avoid_vectors


def avoid_field(o):
    force_vectors = None
    x_boards: list = [4, 6]
    z_boards: list = [6, 8]
    nx = 5
    nz = 5
    force_vectors = avoid_field_internal_func(o, force_vectors, x_boards, z_boards, nx, nz)
    return force_vectors


def draw_vedo_and_save(o, i_time: int, fig_view, camera, app_diagram: bool = True):
    """Функция рисует всё, берёт и возрвращает объект vedo"""
    msh = plot_iterations_new(o).color("silver")
    msh += plot_apps_new(o)
    if o.coordinate_system == 'orbital':
        # msh += fig_plot(o, o.line_str_orf, None)
        # msh += avoid_field(o)  # А вот это ты крутой конечно, но оно кушает много
        for i in range(o.a.n):
            # msh += fig_plot(o, o.line_app_brf[i], o.b_o(o.a.target[i]))
            msh += fig_plot(o, o.line_app_orf[i])  # , color='c')
            msh += fig_plot(o, line_target(r=o.b_o(o.a.target[i]), d=o.d_to_grab), o.b_o(o.a.target[i]), color='c')
        if (not o.survivor) and o.collision_foo == 'Line':
            msh += fig_plot(o, line_chaos(), None)
    elif o.coordinate_system == 'body':
        msh += fig_plot(o, o.line_app_brf, o.a.target[0])
    elif o.coordinate_system == 'real':
        pass
    else:
        raise Exception("Укажите систему координат правильно!")

    fig_view.pop().add(msh)
    # fig_view.fly_to(o.a.r[0])

    n, b, tau = orientation_taken_rod(o, id_app=0)
    if o.coordinate_system == 'real':
        # pos = o.U.T @ o.a.r[0] + o.A.T @ n * 10 - o.A.T @ b * 1
        # view_to = o.U.T @ o.a.r[0] - pos

        pos = o.a.r[0] + o.S.T @ n * 10 - o.S.T @ b * 1
        view_to = - pos + np.array([1, 0, 0])  # o.a.r[0] - pos

        camera.SetPosition(pos)
        # camera.SetRoll(90)
        camera.SetEyePosition(view_to)

    if o.is_saving and (i_time % o.save_rate) == 0:
        o.frame_counter += 1
        filename = 'img/gibbon_' + str('{:04}'.format(o.frame_counter)) + '.png'
        fig_view.screenshot(filename)
        draw_reference_frames(o, 8)
        img = Image.open(filename)
        watermark = Image.open('storage/tmp_pic.png')
        we, hi = watermark.size
        size = 0.29
        img.paste(watermark.crop((int(we * size + 25), int(hi * size + 25),
                                  int(we * (1 - size)), int(hi * (1 - size) - 25))))
        watermark.close()
        if app_diagram:
            watermark = Image.open('storage/tmp_pic2.png').resize((509, 200))
            img.paste(watermark, (0, hi))
            img.save(filename)
            img.close()
            watermark.close()
    return fig_view
