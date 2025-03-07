"""Assembling general problem solution"""
from all_objects import *
"""КОСТЫЛЬ НА ОБЩИЙ ЦЕНТР МАСС"""
"""КОСТЫЛЬ НА МАССУ КОНТЕЙНЕРА"""

choice = '2'
vedo_picture = True
vedo_picsave = False

"""Бестопливный перелёт"""
method = "hkw_analytics"
# method = "2d_analytics"
# method = "diffevolve+shooting"

"""Подходы с применением топлива"""
# method += "+pd"
# method += "+imp"
# method = "const-propulsion"
# method = "linear-propulsion"
# method = "linear-angle

o_global = AllProblemObjects(if_impulse_control=False,
                             if_PID_control=False,
                             if_LQR_control=False,
                             if_avoiding=True,

                             is_saving=vedo_picture and vedo_picsave,
                             save_rate=2,
                             if_talk=False,
                             if_testing_mode=True,
                             choice_complete=False,

                             method=method,
                             if_T_in_shooting=False,
                             begin_rotation='x' if choice == '4' else 'xx',
                             w_twist=1e-4 if choice == '4' else 0.,

                             dt=1., T_max=5500., u_max=0.2 if choice == '4' else 0.05,
                             a_pid_max=1e-5, k_p=3e-4, freetime=50,
                             choice=choice, floor=2, extrafloor=0, d_crash=0.2, d_to_grab=0.5,
                             N_apparatus=1, file_reset=True, coordinate_system=['orbital', 'body', 'real'][0])

o_global.my_print(f"Количество стержней: {o_global.s.n_beams}", mode='c')
if o_global.choice == '4':
    for j in range(50):
        o_global.s.flag[j] = np.array([1, 1])
'''if o_global.choice == '3':
    for j in range(12):
        o_global.s.flag[j] = np.array([1, 1])'''

def iteration_func(o):
    o.time_step()
    o.line_str_orf = np.append(o.line_str_orf, o.r_ub)

    for id_app in o.a.id:
        # Repulsion
        o.a.busy_time[id_app] -= o.dt if o.a.busy_time[id_app] >= 0 else 0
        if (not o.a.flag_fly[id_app]) and o.a.busy_time[id_app] < 0 and False:  # УБРАТЬ ШАМАНСТВО
            u_a_priori = o.get_repulsion(id_app)
            print(f"отталкивание из файла {u_a_priori}")
            u = repulsion(o, id_app, u_a_priori=u_a_priori)
            o.file_save(f'отталкивание {id_app} {u[0]} {u[1]} {u[2]}')
            o.repulsion_save(f'отталкивание {id_app} {u[0]} {u[1]} {u[2]}')

        # Motion control
        o.control_step(id_app)

        # Capturing
        discrepancy = o.get_discrepancy(id_app)
        if (discrepancy < o.d_to_grab) and o.a.flag_fly[id_app]:
            capturing(o=o, id_app=id_app)

        # Docking
        m_a, m_ub = o.get_masses(0)
        tmp = (m_a * o.a.r[0] + m_ub * o.r_ub) / (m_a + m_ub) if o.a.flag_fly else o.r_ub
        o.file_save(f'график {id_app} {discrepancy} {np.linalg.norm(o.w)} '
                    f'{np.linalg.norm(180 / np.pi * np.arccos(clip((np.trace(o.S.T @ o.S_0) - 1) / 2, -1, 1)))} '
                    f'{np.linalg.norm(o.v_ub)} {np.linalg.norm(o.r_ub)} {np.linalg.norm(o.a_self[id_app])} '
                    f'{np.linalg.norm(tmp)} {np.linalg.norm(o.a.r[id_app])}')
        if o.iter % 10 == 0:
            o.line_app_brf[id_app] = np.append(o.line_app_brf[id_app], o.o_b(o.a.r[id_app]))
            o.line_app_orf[id_app] = np.append(o.line_app_orf[id_app], o.a.r[id_app])

        # Stop criteria
        if np.linalg.norm(o.a.r[id_app]) > 1e3:
            o.my_print(f"МИССИЯ ПРОВАЛЕНА! ДОПУЩЕНА ПОТЕРЯ АППАРАТА!", mode='m')
            o.t = 2 * o.T_total
    return o

def iteration_timer(event):
    global o_global, vedo_picture, fig_view, camera
    if o_global.t <= o_global.T_total:
        o_global = iteration_func(o_global)
        if vedo_picture and o_global.iter % o_global.save_rate == 0:
            fig_view = draw_vedo_and_save(o_global, o_global.iter, fig_view, camera, app_diagram=False)

def button_func():
    global timerId
    fig_view.timer_callback("destroy", timerId)
    if "Play" in button.status():
        timerId = fig_view.timer_callback("create")
    button.switch()


if __name__ == "__main__":
    global timerId, fig_view, button, evnetId, camera
    if vedo_picture:
        timerId = 1
        bg = "hdri/6.hdr" if o_global.coordinate_system == 'real' else 'bb'
        fig_view = Plotter(bg='white', size=(1920, 1080))
        button = fig_view.add_button(button_func, states=["Play ", "Pause"], size=20,
                                     font='Bongas', bold=True, pos=[0.98, 0.96])
        # fig_view.timer_callback("destroy", timerId)
        evnetId = fig_view.add_callback("timer", iteration_timer)

        my_mesh = plot_iterations_new(o_global).color("silver")
        app_mesh = plot_apps_new(o_global)

        n, b, tau = orientation_taken_rod(o_global, id_app=0)
        camera = vedo.oriented_camera(center=o_global.a.r[0],
                                      up_vector=o_global.S.T @ n,
                                      backoff_vector=o_global.S.T @ tau) \
            if o_global.coordinate_system == 'real' else None
        fig_view.show(__doc__, my_mesh + app_mesh, zoom=0.5, camera=camera, bg=bg)

    else:
        while True:
            iteration_timer()
