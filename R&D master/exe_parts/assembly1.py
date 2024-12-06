from NIR3 import return_home
from all_objects import *
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import os
global tmp_n
global timerId, fig_view, button
tmp_n = 0


def return_home0():
    from NIR3 import return_home
    global root
    root.destroy()
    return_home()


def return_back0():
    from exe_parts.assembly0 import click_button_assembly
    global root
    root.destroy()
    click_button_assembly()


def button_func_local():
    global o, vedo_pic, label_time_1, label_time_2, canvas2, tmp_n
    global timerId, fig_view, button, start_time, i_time
    global f, vedo_picture
    fig_view.timer_callback("destroy", timerId)
    if "Play" in button.status():
        timerId = fig_view.timer_callback("create", dt=1)
    button.switch()


def iteration_timer(eventId=None):
    from main import iteration_func
    global timerId, button, fig_view
    global o, i_time, start_time, vedo_picture, tmp_n

    if o.is_saving and (i_time % o.save_rate) == 0:
        fileName = "storage/tmp_pic2"  #  + str('{:04}'.format(tmp_n))
        canvas2.postscript(file=fileName + '.eps')
        img = Image.open(fileName + '.eps')
        img.save(fileName + '.png', 'png')
        os.remove(fileName + '.eps')
        tmp_n += 1

    # 3d plotting and saving
    if vedo_picture:
        fig_view = draw_vedo_and_save(o, i_time, fig_view)

    f = open('storage/main.txt', 'a')
    if i_time <= o.T_total/o.dt:  # limitation of time
        o, i_time, f = iteration_func(o, i_time, f)

    o = update_apps(o)
    tm = datetime.now() - start_time
    label_time_1["text"] = f"Время расчётное:                 {print_time((i_time + 1) * o.dt, simple=True)}"
    label_time_2["text"] = f"Время работы программы:  {print_time(tm, simple=False)}"
    f.close()


def run_local():
    global vedo_pic, label_time_1, label_time_2, canvas2, tmp_n
    global timerId, fig_view, button, start_time, i_time
    global o, f, vedo_picture
    f = open('storage/main.txt', 'w')
    f.close()
    # from main import iteration_timer
    start_time = datetime.now()
    vedo_picture = vedo_pic
    i_time = 0
    if vedo_pic:
        timerId = None
        fig_view = Plotter(bg='bb', size=(1920, 1080))
        button = fig_view.addButton(button_func_local, states=["Play ", "Pause"], size=20, font='Bongas', bold=True, pos=[0.9, 0.9])
        fig_view.timer_callback("destroy", timerId)
        evnetId = fig_view.add_callback("timer", iteration_timer)

        my_mesh = plot_iterations_new(o).color("silver")
        app_mesh = plot_apps_new(o)
        fig_view.show(__doc__, my_mesh + app_mesh, zoom=0.5)
    else:
        while True:
            f = open('storage/main.txt', 'a')
            iteration_timer()
            o = update_apps(o)
            tm = datetime.now() - start_time
            label_time_1["text"] = f"Время расчётное:                 {print_time(i_time * o.dt, simple=True)}"
            label_time_2["text"] = f"Время работы программы:  {print_time(tm, simple=False)}"
            f.close()
            if o.is_saving and (i_time % o.save_rate) == 0:
                # canvas2.update()
                fileName = "storage/tmp_pic2_" + str('{:04}'.format(tmp_n))
                canvas2.postscript(file=fileName + '.eps')
                img = Image.open(fileName + '.eps')
                img.save(fileName + '.png', 'png')
                os.remove(fileName + '.eps')
                img.close()
                tmp_n += 1


def update_apps(o1):
    global label11, label12, label13, label14, label15, label16, back_yes, back_no, frame_canvas2, canvas2
    for i in range(o1.N_app):
        label11[i]["background"] = back_yes if o1.X_app.flag_fly[i] == 1 else back_no
        label12[i]["background"] = back_yes if o1.X_app.flag_start[i] == 1 else back_no
        label13[i]["background"] = back_yes if o1.X_app.flag_beam[i] is not None else back_no
        label14[i]["background"] = back_yes if o1.X_app.flag_hkw[i] else back_no
        label15[i]["background"] = back_yes if o1.X_app.busy_time[i] <= 1e-5 else back_no
        label16[i]["background"] = back_yes if o.flag_vision[i] else back_no
        label11[i]["text"] = "Полёт" if o1.X_app.flag_fly[i] == 1 else "Захват"
        label12[i]["text"] = "Старт" if o1.X_app.flag_start[i] == 1 else "Не старт"
        label13[i]["text"] = "Стержень [-]" if o1.X_app.flag_beam[i] is None else f"Стержень [{o1.X_app.flag_beam[i]}]"
        label14[i]["text"] = "Хку" if o1.X_app.flag_hkw[i] else "Шаг по t"
        label15[i]["text"] = "Свободен" if o1.X_app.busy_time[i] <= 1e-5 else "Занят"
        label16[i]["text"] = "Видит цель" if o.flag_vision[i] else "Не видит цель"
    ############################################################################################
    canvas2 = Canvas(frame_canvas2)  # , bg="yellow")
    canvas2.grid(row=0, column=0, sticky="news")
    canvas2.configure()
    frame2 = Frame(canvas2)  # , bg="blue")
    canvas2.create_window((0, 0), window=frame2, anchor='nw')
    ############################################################################################
    total_H = 590
    otstup = 80
    arec = 5
    for i in range(o1.N_app):
        Dr = np.linalg.norm(o.b_o(o.X_app.target[i]) - np.array(o.X_app.r[i]))
        h_tmp = otstup + (total_H - 2 * otstup) / N_app * i
        x_app = special_func(o.X_app.r_0[i], Dr, otstup, o.X_app.flag_start[i], o.X_app.flag_fly[i],
                             o.X_app.flag_beam[i], o.X_app.flag_to_mid[i])
        canvas2.create_line(otstup, h_tmp, 1915 - otstup, h_tmp, width=2, fill="#9AC0CD")
        canvas2.create_oval(x_app - 2 * arec, h_tmp - 2 * arec, x_app + 2 * arec, h_tmp + 2 * arec,
                                       fill="#836FFF",
                                       outline="#473C8B")
        canvas2.create_rectangle(otstup - arec, h_tmp - arec, otstup + arec, h_tmp + arec, fill="#FFB6C1",
                                 outline="#8B5F65")
        canvas2.create_rectangle(1915 - otstup - arec, h_tmp - arec, 1915 - otstup + arec, h_tmp + arec,
                                 fill="#1E90FF",
                                 outline="#483D8B")
        canvas2.create_rectangle(957 - arec, h_tmp - arec, 957 + arec, h_tmp + arec, fill="#DDA0DD",
                                 outline="#8B668B")
    ############################################################################################
    frame2.update_idletasks()
    frame_canvas2.config(width=1915,
                         height=total_H + otstup * 2)
    canvas2.config(scrollregion=canvas2.bbox("all"))
    ############################################################################################
    return o1


def special_func(r_0, Dr, otstup, flag_start, flag_fly, flag_beam, flag_to_mid):
    tmp = 957 - otstup
    if flag_start > 0:
        x_app = otstup
    else:
        if flag_fly > 0:
            if flag_to_mid:
                if flag_beam is not None:  # 1 прямой
                    # x_app = clip(otstup + 957 * Dr / r_0, otstup, otstup + 957)
                    x_app = otstup + tmp * (1 - clip(Dr / r_0, 0, 1))
                else:  # 2 обратный
                    x_app = otstup + tmp * (1 + clip(Dr / r_0, 0, 1))
            else:
                if flag_beam is not None:  # 2 прямой
                    x_app = otstup + tmp * (2 - clip(Dr / r_0, 0, 1))
                else:  # 1 обратный
                    x_app = otstup + tmp * clip(Dr / r_0, 0, 1)
        else:
            if flag_to_mid:
                x_app = 957
            else:
                x_app = 1915 - otstup
    return int(x_app)


def full_assembly():
    global root, o, vedo_pic, canvas2, frame_canvas2, label16
    global N_app, label11, label12, label13, label14, label15, back_yes, back_no, frame2, label_time_1, label_time_2
    # from main import iteration_func, time_is, iteration_timer, button_func, run_it_all
    from main import iteration_timer
    f_tmp = open("storage/params.txt", "r")
    d_crash = None
    for line in f_tmp:
        lst = line.split()
        choice = lst[0]
        control = lst[1]
        vedo_pic = lst[2] == 'True'
        is_saving = lst[3] == 'True'
        if_testing_mode = lst[4] == 'True'
        save_rate = int(lst[5])
        dt = float(lst[6])
        k_p = float(lst[7])
        N_app = int(lst[8])
        T_max = float(lst[9])
        u_max = float(lst[10])
        du_impulse_max = float(lst[11])
        w_max = float(lst[12])
        V_max = float(lst[13])
        R_max = float(lst[14])
        j_max = float(lst[15])
        a_max = float(lst[16])
        d_to_grab = float(lst[17])
        d_crash = float(lst[18])
    f_tmp.close()
    if d_crash is None:
        raise("Файл с параметрами не читается! Иди исправляй!")

    root = Tk()
    root.title("Проект Г.И.Б.О.Н.: сборка")
    root.geometry("1980x1080+0+0")
    root.minsize(300, 200)
    root.maxsize(1980, 1080)
    photo_home = PhotoImage(file="icons/home.png").subsample(10, 10)
    photo_assembly = PhotoImage(file="icons/solution.png").subsample(10, 10)
    photo_back = PhotoImage(file="icons/back.png").subsample(10, 10)
    photo_what = PhotoImage(file="icons/what.png").subsample(10, 10)

    o = AllProblemObjects(choice=choice, if_impulse_control=True if control == '2' else False,
                          if_PID_control=True if control == '3' else False,
                          if_LQR_control=True if control == '4' else False,
                          is_saving=is_saving, if_testing_mode=if_testing_mode, save_rate=save_rate, dt=dt, k_p=k_p,
                          N_apparatus=N_app, T_max=T_max, u_max=u_max, du_impulse_max=du_impulse_max, w_max=w_max,
                          V_max=V_max, R_max=R_max, j_max=j_max, a_pid_max=a_max, d_to_grab=d_to_grab,
                          d_crash=d_crash, if_avoiding=True, if_talk=True)

    ############################################################################################
    frame_canvas = Frame(root)
    frame_canvas.grid(row=3, column=0, columnspan=3, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    frame_canvas.grid_propagate(False)
    canvas = Canvas(frame_canvas)
    canvas.grid(row=0, column=0, sticky="news")
    vsb2 = Scrollbar(frame_canvas, orient="horizontal", command=canvas.xview)
    vsb2.grid(row=1, column=0, sticky='ew')
    canvas.configure(xscrollcommand=vsb2.set)
    frame_buttons = Frame(canvas) 
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')
    ############################################################################################

    btn_home = Button(text="На главную", command=return_home0, image=photo_home, compound=LEFT)
    btn_back = Button(text="Назад", command=return_back0, image=photo_back, compound=LEFT)
    btn_run = Button(text="Начать расчёт", command=run_local, image=photo_what, compound=LEFT)
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=EW)
    btn_back.grid(row=0, column=1, padx='7', pady='7', sticky=EW)
    btn_run.grid(row=0, column=2, padx='7', pady='7', sticky=EW)

    label_time_1 = ttk.Label(text=f"Время расчётное:                 {print_time(0, simple=True)}")
    label_time_2 = ttk.Label(text=f"Время работы программы:  {print_time(0, simple=True)}")
    label_time_1.grid(row=1, column=0, padx='7', pady='7', sticky=W)
    label_time_2.grid(row=2, column=0, padx='7', pady='7', sticky=W)

    row_count = 0
    # back_yes = "#98F5FF"
    back_yes = "#1E90FF"
    back_no = "#8B5F65"
    label1 = [ttk.Label(frame_buttons, text=f"Аппарат [{i}]", background="#828282", foreground="#E0EEE0", padding=8, width=20) for i in range(N_app)]
    label11 = [ttk.Label(frame_buttons, text="Полёт", background="#9E9E9E", foreground="#E0EEE0", padding=8) for i in range(N_app)]
    label12 = [ttk.Label(frame_buttons, text="Старт", background="#9E9E9E", foreground="#E0EEE0", padding=8) for i in range(N_app)]
    label13 = [ttk.Label(frame_buttons, text="Стержень", background="#9E9E9E", foreground="#E0EEE0", padding=8) for i in range(N_app)]
    label14 = [ttk.Label(frame_buttons, text="Хку", background="#9E9E9E", foreground="#E0EEE0", padding=8) for i in range(N_app)]
    label15 = [ttk.Label(frame_buttons, text="Cвободен", background="#9E9E9E", foreground="#E0EEE0", padding=8) for i in range(N_app)]
    label16 = [ttk.Label(frame_buttons, text="Не видит", background="#9E9E9E", foreground="#E0EEE0", padding=8) for i in range(N_app)]
    for i in range(N_app):
        label11[i]["background"] = back_yes if o.X_app.flag_fly[i] else back_no
        label12[i]["background"] = back_yes if o.X_app.flag_start[i] else back_no
        label13[i]["background"] = back_yes if o.X_app.flag_beam[i] is not None else back_no
        label14[i]["background"] = back_yes if o.X_app.flag_hkw[i] else back_no
        label15[i]["background"] = back_yes if o.X_app.busy_time[i] <= 1e-5 else back_no
        label16[i]["background"] = back_yes if o.flag_vision[i] else back_no
        label11[i]["text"] = "Полёт" if o.X_app.flag_fly[i] else "Захват"
        label12[i]["text"] = "Старт" if o.X_app.flag_start[i] else "Не старт"
        label13[i]["text"] = "Стержень [-]" if o.X_app.flag_beam[i] is None else f"Стержень [{o.X_app.flag_beam[i]}]"
        label14[i]["text"] = "Хку" if o.X_app.flag_hkw[i] else "Шаг по t"
        label15[i]["text"] = "Свободен" if o.X_app.busy_time[i] <= 1e-5 else "Занят"
        label16[i]["text"] = "Видит цель" if o.flag_vision[i] else "Не видит цель"
        label1[i].grid(row=row_count, column=i, padx='7', pady='7', sticky=EW)
        label11[i].grid(row=row_count+1, column=i, padx='7', pady='7', sticky=EW)
        label12[i].grid(row=row_count+2, column=i, padx='7', pady='7', sticky=EW)
        label13[i].grid(row=row_count+3, column=i, padx='7', pady='7', sticky=EW)
        label14[i].grid(row=row_count+4, column=i, padx='7', pady='7', sticky=EW)
        label15[i].grid(row=row_count+5, column=i, padx='7', pady='7', sticky=EW)
        label16[i].grid(row=row_count+6, column=i, padx='7', pady='7', sticky=EW)

    frame_buttons.update_idletasks()
    frame_canvas.config(width=1915,
                        height=400)
    canvas.config(scrollregion=canvas.bbox("all"))

    ############################################################################################
    frame_canvas2 = Frame(root)
    frame_canvas2.grid(row=5, column=0, columnspan=3, pady=(5, 0), sticky='nw')
    frame_canvas2.grid_rowconfigure(0, weight=1)
    frame_canvas2.grid_columnconfigure(0, weight=1)
    frame_canvas2.grid_propagate(False)
    canvas2 = Canvas(frame_canvas2)  # , bg="yellow")
    canvas2.grid(row=0, column=0, sticky="news")
    canvas2.configure()  # (xscrollcommand=vsb2.set, yscrollcommand=vsb.set)
    frame2 = Frame(canvas2)  # , bg="blue")
    canvas2.create_window((0, 0), window=frame2, anchor='nw')
    ############################################################################################
    total_H = 400
    otstup = 80
    arec = 5
    for i in range(N_app):
        Dr = np.linalg.norm(o.b_o(o.X_app.target[i])- np.array(o.X_app.r[i]))
        h_tmp = otstup + (total_H - 2 * otstup) / N_app * i
        x_app = special_func(o.X_app.r_0[i], Dr, otstup, o.X_app.flag_start[i], o.X_app.flag_fly[i],
                             o.X_app.flag_beam[i], o.X_app.flag_to_mid[i])

        canvas2.create_line(otstup, h_tmp, 1915-otstup, h_tmp, width=2, fill="#9AC0CD")
        canvas2.create_oval(x_app-2*arec, h_tmp-2*arec, x_app+2*arec, h_tmp+2*arec, fill="#836FFF",
                                       outline="#473C8B")
        canvas2.create_rectangle(otstup-arec, h_tmp-arec, otstup+arec, h_tmp+arec, fill="#FFB6C1", outline="#8B5F65")
        canvas2.create_rectangle(1915-otstup-arec, h_tmp-arec, 1915-otstup+arec, h_tmp+arec, fill="#1E90FF",
                                 outline="#483D8B")
        canvas2.create_rectangle(957-arec, h_tmp-arec, 957+arec, h_tmp+arec, fill="#DDA0DD",
                                 outline="#8B668B")

    ############################################################################################
    frame2.update_idletasks()
    frame_canvas2.config(width=1915,
                        height=total_H+otstup*2)
    canvas2.config(scrollregion=canvas2.bbox("all"))
    ############################################################################################

    root.mainloop()
