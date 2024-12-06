from main_interface import *
from tkinter_files.tk_functions import *

# Common params
choice = '3'
control = '3'
vedo_pic = True
is_saving = False
if_testing_mode = False
save_rate = 5
dt = 10.0
N_app = 2

# Control params
k_p = 1e-4
a_max = 0.0002
du_impulse_max = 0.4

# Constrains
d_to_grab = 0.5
d_crash = 0.1
T_max = 400.
u_max = 0.2
w_max = 0.0015
V_max = 0.1
R_max = 9.
j_max = 30.

# Extra
entry3_state = 'DISABLED'
entry4_state = 'DISABLED'
entry5_state = 'DISABLED'
entry6_state = 'DISABLED'
o_global = AllProblemObjects()

def change_check_1():
    global check_var_1, check_label_1, vedo_pic
    vedo_pic = True if check_var_1.get() > 0 else False
    check_label_1["text"] = check_var_1.get()

def change_check_2():
    global check_var_2, check_label_2, is_saving, entry3_state, label_3, save_rate
    is_saving = True if check_var_2.get() > 0 else False
    check_label_2["text"] = check_var_2.get()
    entry3_state = 'NORMAL' if is_saving else 'DISABLED'
    label_3["text"] = f"{save_rate} итераций" if entry3_state == 'NORMAL' else "[изображения не сохраняются]"

def change_check_3():
    global check_var_3, check_label_3, if_testing_mode
    if_testing_mode = True if check_var_3.get() > 0 else False
    check_label_3["text"] = check_var_3.get()

def return_home0(event=None):
    from main_interface import return_home
    global root
    root.destroy()
    return_home()

def full_assembly0():
    from tkinter_files.assembly1 import full_assembly
    global choice, control, vedo_pic, is_saving, if_testing_mode, save_rate, dt, k_p, N_app, T_max
    global u_max, du_impulse_max, w_max, V_max, R_max, j_max, a_max, d_to_grab, d_crash
    f_tmp = open('storage/params.txt', 'w')
    f_tmp.write(f"{choice} {control} {vedo_pic} {is_saving} {if_testing_mode} {save_rate} {dt} {k_p} {N_app} {T_max} "
                f"{u_max} {du_impulse_max} {w_max} {V_max} {R_max} {j_max} {a_max} {d_to_grab} {d_crash} \n")
    f_tmp.close()
    root.destroy()
    full_assembly()

def save_params():
    global choice, control, vedo_pic, is_saving, if_testing_mode, save_rate, dt, k_p, N_app, T_max
    global u_max, du_impulse_max, w_max, V_max, R_max, j_max, a_max, d_to_grab, d_crash
    f_tmp = open('storage/saved_params.txt', 'w')
    f_tmp.write(f"{choice} {control} {vedo_pic} {is_saving} {if_testing_mode} {save_rate} {dt} {k_p} {N_app} {T_max} "
                f"{u_max} {du_impulse_max} {w_max} {V_max} {R_max} {j_max} {a_max} {d_to_grab} {d_crash} \n")
    f_tmp.close()

def download_params():
    global root, label_1, entry_1, label_2, entry_2, b_app, label_3, entry_3, entry3_state, choice_1_vars
    global label_4, entry_4, entry4_state, label_5, entry_5, entry5_state, entry6_state
    global label_6, entry_6, label_7, entry_7, label_8, entry_8, label_9, entry_9, label_10, entry_10
    global label_11, entry_11, label_12, entry_12, photo_app_1, photo_plus, img_label_1, photo_consts
    global check_var_1, check_label_1, check_var_2, check_label_2, check_var_3, check_label_3
    global choice, control, vedo_pic, is_saving, if_testing_mode, save_rate, dt, k_p, N_app, d_crash
    global T_max, t_reaction, time_to_be_busy, u_max, du_impulse_max, w_max, V_max, R_max, j_max, a_max, d_to_grab
    global label_choice_extra_1, choice_1, choice_2_vars, label_choice_extra_2, choice_2, o_global
    global label_1, entry_1, dt
    f_tmp = open('storage/saved_params.txt', 'r')
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
    check_label_1["text"] = check_var_1.get()
    check_label_2["text"] = check_var_2.get()
    entry3_state = 'NORMAL' if is_saving else 'DISABLED'
    label_3["text"] = f"{save_rate} итераций" if entry3_state == 'NORMAL' else "[изображения не сохраняются]"
    check_label_3["text"] = check_var_3.get()
    label_1["text"] = f"{dt} секунд"
    label_2["text"] = f"{N_app} аппарат{okonchanye(N_app)}"
    photo_app_n = merge_n_photos(photo_app_1, photo_plus, N_app)
    img = ImageTk.PhotoImage(photo_app_n)
    b_app.configure(image=img)
    b_app.image = img
    label_3["text"] = f"{save_rate} итераций"
    label_4["text"] = f"{k_p}"
    label_5["text"] = f"{float(a_max * 1e3)} мН"
    label_6["text"] = f"{du_impulse_max} м/с"
    label_7["text"] = f"{int(100 * d_crash)} см"
    label_8["text"] = f"{int(d_to_grab * 100)} см"
    label_9["text"] = f"{T_max} секунд, {int(100 * T_max / (2 * np.pi / o_global.w_hkw))}% оборота"
    label_10["text"] = f"{u_max * 100} см/с"
    label_11["text"] = f"{w_max} рад/с, оборот раз в {2 * np.pi / w_max} секунд"
    label_12["text"] = f"{j_max} градусов"
    label_choice_extra_1.config(text=f"{choice_1_vars[int(choice) - 1]}")
    img1 = ImageTk.PhotoImage(photo_consts[int(choice)-1])
    img_label_1.configure(image=img1)
    img_label_1.image = img1
    if control == '2':
        entry6_state = 'NORMAL'
        label_6["text"] = f"{du_impulse_max} м/с"
    else:
        entry6_state = 'DISABLED'
        label_6["text"] = f"[нет импульсного управления]"
    if control == '3':
        entry4_state = 'NORMAL'
        label_4["text"] = f"{k_p}"
    else:
        entry4_state = 'DISABLED'
        label_4["text"] = f"[не управляется ПД-регулятором]"
    if (control == '3') or (control == '4'):
        entry5_state = 'NORMAL'
        label_5["text"] = f"{int(a_max * 1e3)} мН"
    else:
        entry5_state = 'DISABLED'
        label_5["text"] = f"[нет непрерывного управления]"
    label_choice_extra_2.config(text=f"{choice_2_vars[int(control) - 1]}")

def change_entry_1():
    global label_1, entry_1, dt
    dt = float(entry_1.get())
    label_1["text"] = f"{dt} секунд"

def change_entry_2():
    global label_2, entry_2, N_app, b_app, icons
    N_app = int(entry_2.get())
    label_2["text"] = f"{N_app} аппарат{okonchanye(N_app)}"
    icons.app_n = merge_n_photos(icons.app_1, icons.plus, N_app)
    img = ImageTk.PhotoImage(icons.app_n)
    b_app.configure(image=img)
    b_app.image = img

def change_entry_3():
    global label_3, entry_3, save_rate, entry3_state
    save_rate = int(entry_3.get())
    label_3["text"] = f"{save_rate} итераций" if entry3_state == 'NORMAL' else "[изображения не сохраняются]"

def change_entry_4():
    global label_4, entry_4, k_p
    k_p = float(entry_4.get())
    label_4["text"] = f"{k_p}"

def change_entry_5():
    global label_5, entry_5, a_max
    a_max = float(entry_5.get())
    label_5["text"] = f"{float(a_max * 1e3)} мН"


def change_entry_6():
    global label_6, entry_6, du_impulse_max
    du_impulse_max = float(entry_6.get())
    label_6["text"] = f"{du_impulse_max} м/с"

def change_entry_7():
    global label_7, entry_7, d_crash
    d_crash = float(entry_7.get())
    label_7["text"] = f"{int(100 * d_crash)} см"

def change_entry_8():
    global label_8, entry_8, d_to_grab
    d_to_grab = float(entry_8.get())
    label_8["text"] = f"{int(100 * d_to_grab)} см"

def change_entry_9():
    global label_9, entry_9, T_max, o_global
    T_max = float(entry_9.get())
    label_9["text"] = f"{T_max} секунд, {int(100 * T_max / (2 * np.pi / o_global.w_hkw))}% оборота"


def change_entry_10():
    global label_10, entry_10, u_max
    u_max = float(entry_10.get())
    label_10["text"] = f"{u_max * 100} см/с"


def change_entry_11():
    global label_11, entry_11, w_max
    w_max = float(entry_11.get())
    label_11["text"] = f"{w_max} рад/с, оборот раз в {2 * np.pi / w_max} секунд"


def change_entry_12():
    global label_12, entry_12, j_max
    j_max = float(entry_12.get())
    label_12["text"] = f"{j_max} градусов"


def choice_const():
    global label_choice_extra_1, choice, choice_1, img_label_1, photo_consts, choice_1_vars
    choice = choice_1.get()
    label_choice_extra_1.config(text=f"{choice_1_vars[int(choice) - 1]}")
    img1 = ImageTk.PhotoImage(photo_consts[int(choice)-1])
    img_label_1.configure(image=img1)
    img_label_1.image = img1


def choice_control(change_other=True):
    global choice_2_vars, label_choice_extra_2, choice_2, control, entry4_state, k_p, label_4
    global entry5_state, a_max, label_5, entry6_state, du_impulse_max, label_6
    control = choice_2.get()
    entry6_state = 'NORMAL' if control == '2' else 'DISABLED'
    entry4_state = 'NORMAL' if control == '3' else 'DISABLED'
    entry5_state = 'NORMAL' if (control == '3') or (control == '4') else 'DISABLED'
    label_choice_extra_2.config(text=f"{choice_2_vars[int(control) - 1]}")
    if change_other:
        label_6["text"] = f"{du_impulse_max} м/с" if control == '2' else f"[нет импульсного управления]"
        label_4["text"] = f"{k_p}" if control == '3' else f"[не управляется ПД-регулятором]"
        label_5["text"] = f"{int(a_max * 1e3)} мН" if (control == '3') or (control == '4') else f"[нет непрерывного управления]"


def merge_n_photos(photo_1, photo_plus, N, limit=10):
    tmp = 1+0.5*(N-1) if N < 10 else 1+0.5*(limit-1)
    if N > limit:
        photo_n = Image.new('RGBA', (int((tmp+1) * photo_1.size[0]), photo_1.size[1]), (218, 218, 218))
    else:
        photo_n = Image.new('RGBA', (int(tmp*photo_1.size[0]), photo_1.size[1]), (218, 218, 218))
    for i in range(N):
        if i < limit:
            photo_n.paste(photo_1, (int(i*photo_1.size[0]/2), 0), mask=photo_1)
    if N > limit:
        photo_n.paste(photo_plus, (int(limit * photo_1.size[0] / 2), int((photo_1.size[1] - photo_plus.size[1])/2)),
                      mask=photo_plus)
    return photo_n


def click_button_assembly():
    global root, label_1, entry_1, label_2, entry_2, b_app, label_3, entry_3, entry3_state, choice_1_vars
    global label_4, entry_4, entry4_state, label_5, entry_5, entry5_state, entry6_state
    global label_6, entry_6, label_7, entry_7, label_8, entry_8, label_9, entry_9, label_10, entry_10
    global label_11, entry_11, label_12, entry_12, photo_app_1, photo_plus, img_label_1, photo_consts
    global check_var_1, check_label_1, check_var_2, check_label_2, check_var_3, check_label_3
    global choice, control, vedo_pic, is_saving, if_testing_mode, save_rate, dt, k_p, N_app, T_max, t_reaction
    global time_to_be_busy, u_max, du_impulse_max, w_max, V_max, R_max, j_max, a_max, d_to_grab, d_crash
    global label_choice_extra_1, choice_1, choice_2_vars, label_choice_extra_2, choice_2, o_global, icons
    o = AllProblemObjects()
    root = Tk()
    icons = Icons()
    root.title("Проект Г.И.Б.О.Н.: сборка")
    root.geometry("1980x1080+0+0")
    root.minsize(1000, 685)
    root.maxsize(1980, 1080)

    btn_home = Button(text="На главную", command=return_home0, image=icons.home, compound=LEFT)
    btn_next = Button(text="Далее", command=full_assembly0, image=icons.next, compound=LEFT)
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=EW)
    btn_next.grid(row=0, column=1, padx='7', pady='7', sticky=EW)
    root.bind("h", return_home0)

    frame_canvas = Frame(root)
    frame_canvas.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    frame_canvas.grid_propagate(False)
    canvas = Canvas(frame_canvas)
    canvas.grid(row=0, column=0, sticky="news")
    vsb = Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)
    frame = Frame(canvas)  # , bg="blue")
    canvas.create_window((0, 0), window=frame, anchor='nw')

    row = 0
    btn_save = Button(frame, text="Сохранить", command=save_params, image=icons.save, compound=LEFT)
    btn_down = Button(frame, text="Загрузить", command=download_params, image=icons.down, compound=LEFT)
    btn_save.grid(row=row, column=4, padx='7', pady='7', sticky=EW)
    btn_down.grid(row=row, column=5, padx='7', pady='7', sticky=EW)
    row, headline_1 = create_label("Проверьте входные данные:", row, frame)

    row, check_var_1, checkbutton_1, check_label_1 = create_check("Vedo", vedo_pic, row, change_check_1, frame)
    row, check_var_2, checkbutton_2, check_label_2 = create_check("Сохранение", is_saving, row, change_check_2, frame)
    row, check_var_3, checkbutton_3, check_label_3 = create_check("Отладка", if_testing_mode, row, change_check_3, frame)

    row, label_1, entry_1 = create_entry("Шаг по времени", dt, row, change_entry_1, frame)
    change_entry_1()

    b_app = Label(frame, image=ImageTk.PhotoImage(icons.app_1))
    b_app.grid(row=row, column=4, columnspan=4, sticky=W)
    row, label_2, entry_2 = create_entry("Количество аппаратов", N_app, row, change_entry_2, frame)
    change_entry_2()

    row, label_3, entry_3 = create_entry("Сохранение раз в", save_rate, row, change_entry_3, frame)
    change_entry_3()

    choice_1_vars = ['пробная\n(турист)', 'длинная\n(новичок)', 'антенна\n(мастер)', 'станция\n(Сэм)']
    photo_consts = [Image.open("icons/const_1.png"), Image.open("icons/const_2.png"),
                    Image.open("icons/const_3.png"), Image.open("icons/const_4.png")]
    photo_consts = [photo_consts[0].resize((230, 150)), photo_consts[1].resize((230, 150)),
                    photo_consts[2].resize((230, 150)), photo_consts[3].resize((230, 150))]

    row, choice_1, label_choice_1, label_choice_extra_1, img_label_1 = create_choice("Выбор конструкции:", choice, row,
                                                                                     4, choice_const, frame)
    choice_const()
    choice_2_vars = ['без управления', 'импульсное', 'ПД-регулятор', 'ЛКР']
    row, choice_2, label_choice_2, label_choice_extra_2, img_label_2 = create_choice("Выбор управления:", control, row,
                                                                                     4, choice_control, frame)
    choice_control(change_other=False)

    row, headline_2 = create_label("Параметры управления", row, frame)
    row, label_4, entry_4 = create_entry("Коэффициент ПДР", k_p, row, change_entry_4, frame)
    change_entry_4()
    row, label_5, entry_5 = create_entry("Ускорение двигателя", a_max, row, change_entry_5, frame)
    change_entry_5()
    row, label_6, entry_6 = create_entry("Импульс двигателя", du_impulse_max, row, change_entry_6, frame)
    change_entry_6()

    row, headline_2 = create_label("Ограничения", row, frame)
    row, label_7, entry_7 = create_entry("Радиус опасной зоны", d_crash, row, change_entry_7, frame)
    change_entry_7()
    row, label_8, entry_8 = create_entry("Радиус захвата", d_to_grab, row, change_entry_8, frame)
    change_entry_8()
    row, label_9, entry_9 = create_entry("Время эпизода", T_max, row, change_entry_9, frame)
    change_entry_9()
    row, label_10, entry_10 = create_entry("Скорость отталкивания", u_max, row, change_entry_10, frame)
    change_entry_10()
    row, label_11, entry_11 = create_entry("Угловая скорость", w_max, row, change_entry_11, frame)
    change_entry_11()
    row, label_12, entry_12 = create_entry("Отклонение станции", j_max, row, change_entry_12, frame)
    change_entry_12()

    frame.update_idletasks()
    frame_canvas.config(width=1915,
                        height=920)
    canvas.config(scrollregion=canvas.bbox("all"))

    root.focus_force()
    root.mainloop()
