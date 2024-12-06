from NIR3 import return_home
from all_objects import *
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
global choice, control, vedo_pic, is_saving, if_testing_mode, save_rate, dt, k_p, N_app, T_max, entry3_state, \
    t_reaction, time_to_be_busy, u_max, du_impulse_max, w_max, V_max, R_max, j_max, a_max, d_to_grab, d_crash, \
        entry4_state, entry5_state, entry6_state, o
# Стандартные параметры
choice = '3'  #
control = '3'  #
vedo_pic = True  #
is_saving = False  #
if_testing_mode = False  #
save_rate = 5  #
dt = 10.0  #
N_app = 2  #
# Параметры управления
k_p = 1e-4  #
a_max = 0.0002  #
du_impulse_max = 0.4  #
# Ограничения
d_to_grab = 0.5
d_crash = 0.1
T_max = 400.
u_max = 0.2
w_max = 0.0015
V_max = 0.1
R_max = 9.
j_max = 30.
# Вспомогательное
entry3_state = 'DISABLED'
entry4_state = 'DISABLED'
entry5_state = 'DISABLED'
entry6_state = 'DISABLED'
o = AllProblemObjects()


class Window(Tk):
    def __init__(self):
        super().__init__()
        self.title("Проект Г.И.Б.О.Н.")
        self.geometry("1000x685+200+100")


def checkbutton_changed1():
    global enabled1, enabled_label1, vedo_pic
    vedo_pic = True if enabled1.get() > 0 else False
    enabled_label1["text"] = enabled1.get()


def checkbutton_changed2():
    global enabled2, enabled_label2, is_saving, entry3_state, label3, save_rate
    is_saving = True if enabled2.get() > 0 else False
    enabled_label2["text"] = enabled2.get()
    entry3_state = 'NORMAL' if is_saving else 'DISABLED'
    if entry3_state == 'NORMAL':
        label3["text"] = f"{save_rate} итераций"
    else:
        label3["text"] = f"[изображения не сохраняются]"


def checkbutton_changed3():
    global enabled3, enabled_label3, if_testing_mode
    if_testing_mode = True if enabled3.get() > 0 else False
    enabled_label3["text"] = enabled3.get()


def return_home0():
    from NIR3 import return_home
    global root
    root.destroy()
    return_home()


def full_assembly0():
    from exe_parts.assembly1 import full_assembly
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
    global root, label1, entry1, label2, entry2, b_app, label3, entry3, entry3_state, choice_1_vars
    global label4, entry4, entry4_state, label5, entry5, entry5_state, entry6_state
    global label6, entry6, label7, entry7, label8, entry8, label9, entry9, label10, entry10
    global label11, entry11, label12, entry12, photo_app_1, photo_plus, b_const, photo_consts
    global enabled1, enabled_label1, enabled2, enabled_label2, enabled3, enabled_label3
    global choice, control, vedo_pic, is_saving, if_testing_mode, save_rate, dt, k_p, N_app, d_crash
    global T_max, t_reaction, time_to_be_busy, u_max, du_impulse_max, w_max, V_max, R_max, j_max, a_max, d_to_grab
    global label_choice_1_post, choice_1, choice_2_vars, label_choice_2_post, choice_2, o
    global label1, entry1, dt
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
    enabled_label1["text"] = enabled1.get()
    enabled_label2["text"] = enabled2.get()
    entry3_state = 'NORMAL' if is_saving else 'DISABLED'
    if entry3_state == 'NORMAL':
        label3["text"] = f"{save_rate} итераций"
    else:
        label3["text"] = f"[изображения не сохраняются]"
    enabled_label3["text"] = enabled3.get()
    label1["text"] = f"{dt} секунд"
    label2["text"] = f"{N_app} аппарат{okonchanye(N_app)}"
    photo_app_n = merge_n_photos(photo_app_1, photo_plus, N_app)
    img = ImageTk.PhotoImage(photo_app_n)
    b_app.configure(image=img)
    b_app.image = img
    label3["text"] = f"{save_rate} итераций"
    label4["text"] = f"{k_p}"
    label5["text"] = f"{float(a_max*1e3)} мН"
    label6["text"] = f"{du_impulse_max} м/с"
    label7["text"] = f"{int(100*d_crash)} см"
    label8["text"] = f"{int(d_to_grab*100)} см"
    label9["text"] = f"{T_max} секунд, {int(100*T_max/(2*np.pi/o.w_hkw))}% оборота"
    label10["text"] = f"{u_max*100} см/с"
    label11["text"] = f"{w_max} рад/с, оборот раз в {2*np.pi/w_max} секунд"
    label12["text"] = f"{j_max} градусов"
    label_choice_1_post.config(text=f"{choice_1_vars[int(choice)-1]}")
    img1 = ImageTk.PhotoImage(photo_consts[int(choice)-1])
    b_const.configure(image=img1)
    b_const.image = img1
    if control == '2':
        entry6_state = 'NORMAL'
        label6["text"] = f"{du_impulse_max} м/с"
    else:
        entry6_state = 'DISABLED'
        label6["text"] = f"[нет импульсного управления]"
    if control == '3':
        entry4_state = 'NORMAL'
        label4["text"] = f"{k_p}"
    else:
        entry4_state = 'DISABLED'
        label4["text"] = f"[не управляется ПД-регулятором]"
    if (control == '3') or (control == '4'):
        entry5_state = 'NORMAL'
        label5["text"] = f"{int(a_max*1e3)} мН"
    else:
        entry5_state = 'DISABLED'
        label5["text"] = f"[нет непрерывного управления]"
    label_choice_2_post.config(text=f"{choice_2_vars[int(control)-1]}")


def show_message_1():
    global label1, entry1, dt
    dt = float(entry1.get())
    label1["text"] = f"{dt} секунд"


def okonchanye(N):
    if (N % 10) == 1:
        return ""
    if ((N % 10) > 1) and ((N % 10) < 5):
        return "а"
    if ((N % 10) >= 5) or ((N > 9) and (N < 21)):
        return "ов"


def show_message_2():
    global label2, entry2, N_app, b_app, photo_app_1, photo_plus
    N_app = int(entry2.get())
    label2["text"] = f"{N_app} аппарат{okonchanye(N_app)}"
    photo_app_n = merge_n_photos(photo_app_1, photo_plus, N_app)
    img = ImageTk.PhotoImage(photo_app_n)
    b_app.configure(image=img)
    b_app.image = img


def show_message_3():
    global label3, entry3, save_rate
    save_rate = int(entry3.get())
    label3["text"] = f"{save_rate} итераций"


def show_message_4():
    global label4, entry4, k_p
    k_p = float(entry4.get())
    label4["text"] = f"{k_p}"


def show_message_5():
    global label5, entry5, a_max
    a_max = float(entry5.get())
    label5["text"] = f"{float(a_max*1e3)} мН"


def show_message_6():
    global label6, entry6, du_impulse_max
    du_impulse_max = float(entry6.get())
    label6["text"] = f"{du_impulse_max} м/с"


def show_message_7():
    global label7, entry7, d_crash
    d_crash = float(entry7.get())
    label7["text"] = f"{int(100*d_crash)} см"


def show_message_8():
    global label8, entry8, d_to_grab
    d_to_grab = float(entry8.get())
    label8["text"] = f"{int(d_to_grab*100)} см"


def show_message_9():
    global label9, entry9, T_max, o
    T_max = float(entry9.get())
    label9["text"] = f"{T_max} секунд, {int(100*T_max/(2*np.pi/o.w_hkw))}% оборота"


def show_message_10():
    global label10, entry10, u_max
    u_max = float(entry10.get())
    label10["text"] = f"{u_max*100} см/с"


def show_message_11():
    global label11, entry11, w_max
    w_max = float(entry11.get())
    label11["text"] = f"{w_max} рад/с, оборот раз в {2*np.pi/w_max} секунд"


def show_message_12():
    global label12, entry12, j_max
    j_max = float(entry12.get())
    label12["text"] = f"{j_max} градусов"


def choice_const():
    global label_choice_1_post, choice, choice_1, b_const, photo_consts, choice_1_vars
    choice = choice_1.get()
    label_choice_1_post.config(text=f"{choice_1_vars[int(choice)-1]}")
    img1 = ImageTk.PhotoImage(photo_consts[int(choice)-1])
    b_const.configure(image=img1)
    b_const.image = img1


def choice_control():
    global choice_2_vars, label_choice_2_post, choice_2, control, entry4_state, k_p, label4
    global entry5_state, a_max, label5, entry6_state, du_impulse_max, label6
    control = choice_2.get()
    if control == '2':
        entry6_state = 'NORMAL'
        label6["text"] = f"{du_impulse_max} м/с"
    else:
        entry6_state = 'DISABLED'
        label6["text"] = f"[нет импульсного управления]"
    if control == '3':
        entry4_state = 'NORMAL'
        label4["text"] = f"{k_p}"
    else:
        entry4_state = 'DISABLED'
        label4["text"] = f"[не управляется ПД-регулятором]"
    if (control == '3') or (control == '4'):
        entry5_state = 'NORMAL'
        label5["text"] = f"{int(a_max*1e3)} мН"
    else:
        entry5_state = 'DISABLED'
        label5["text"] = f"[нет непрерывного управления]"
    label_choice_2_post.config(text=f"{choice_2_vars[int(control)-1]}")


def merge_n_photos(photo_1, photo_plus, N, limit=10):
    if N < 5:
        tmp = 1+0.5*(N-1)
    else:
        tmp = 1+0.5*(limit-1)
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
    from tkinter import ttk
    global root, label1, entry1, label2, entry2, b_app, label3, entry3, entry3_state, choice_1_vars
    global label4, entry4, entry4_state, label5, entry5, entry5_state, entry6_state
    global label6, entry6, label7, entry7, label8, entry8, label9, entry9, label10, entry10
    global label11, entry11, label12, entry12, photo_app_1, photo_plus, b_const, photo_consts
    global enabled1, enabled_label1, enabled2, enabled_label2, enabled3, enabled_label3
    global choice, control, vedo_pic, is_saving, if_testing_mode, save_rate, dt, k_p, N_app, \
        T_max, t_reaction, time_to_be_busy, u_max, du_impulse_max, w_max, V_max, R_max, j_max, a_max, d_to_grab, d_crash
    global label_choice_1_post, choice_1, choice_2_vars, label_choice_2_post, choice_2, o
    o = AllProblemObjects()
    root = Tk()
    root.title("Проект Г.И.Б.О.Н.: сборка")
    root.geometry("1980x1080+0+0")
    root.minsize(1000, 685)
    root.maxsize(1980, 1080)
    photo_home = PhotoImage(file="icons/home.png").subsample(10, 10)
    photo_assembly = PhotoImage(file="icons/solution.png").subsample(10, 10)
    photo_what = PhotoImage(file="icons/what.png").subsample(10, 10)
    photo_save = PhotoImage(file="icons/save.png").subsample(10, 10)
    photo_down = PhotoImage(file="icons/download.png").subsample(10, 10)
    photo_next = PhotoImage(file="icons/next.png").subsample(10, 10)

    btn_home = Button(text="На главную", command=return_home0, image=photo_home, compound=LEFT)
    btn_next = Button(text="Далее", command=full_assembly0, image=photo_next, compound=LEFT)
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=EW)
    btn_next.grid(row=0, column=1, padx='7', pady='7', sticky=EW)

    ############################################################################################
    frame_canvas = Frame(root)
    frame_canvas.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    frame_canvas.grid_propagate(False)

    # Add a canvas in that frame
    canvas = Canvas(frame_canvas)  # , bg="yellow")
    canvas.grid(row=0, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)

    # Create a frame to contain the buttons
    frame_buttons = Frame(canvas)  # , bg="blue")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')
    ############################################################################################
    row_count = 0
    label = ttk.Label(frame_buttons, text="Проверьте входные данные:", background="#828282", foreground="#E0EEE0", padding=8)
    btn_save = Button(frame_buttons, text="Сохранить", command=save_params, image=photo_save, compound=LEFT)
    btn_down = Button(frame_buttons, text="Загрузить", command=download_params, image=photo_down, compound=LEFT)
    label.grid(row=row_count, column=0, padx='7', pady='7', columnspan=4, sticky=EW)
    btn_save.grid(row=row_count, column=4, padx='7', pady='7', sticky=EW)
    btn_down.grid(row=row_count, column=5, padx='7', pady='7', sticky=EW)

    row_count += 1
    enabled1 = IntVar()
    enabled1.set(vedo_pic)
    enabled_checkbutton1 = ttk.Checkbutton(frame_buttons, text="Vedo", variable=enabled1, command=checkbutton_changed1)
    vedo_pic = True if enabled1.get() > 0 else False
    enabled_label1 = ttk.Label(frame_buttons, text=vedo_pic)
    enabled_checkbutton1.grid(row=row_count, column=0, padx='7', pady='7')
    enabled_label1.grid(row=row_count, column=1, padx='7', pady='7')

    row_count += 1
    enabled2 = IntVar()
    enabled2.set(is_saving)
    enabled_checkbutton2 = ttk.Checkbutton(frame_buttons, text="Сохранение", variable=enabled2, command=checkbutton_changed2)
    is_saving = True if enabled2.get() > 0 else False
    enabled_label2 = ttk.Label(frame_buttons, text=is_saving)
    enabled_checkbutton2.grid(row=row_count, column=0, padx='7', pady='7')
    enabled_label2.grid(row=row_count, column=1, padx='7', pady='7')

    row_count += 1
    enabled3 = IntVar()
    enabled3.set(if_testing_mode)
    enabled_checkbutton3 = ttk.Checkbutton(frame_buttons, text="Отладка", variable=enabled3, command=checkbutton_changed3)
    if_testing_mode = True if enabled3.get() > 0 else False
    enabled_label3 = ttk.Label(frame_buttons, text=if_testing_mode)
    enabled_checkbutton3.grid(row=row_count, column=0, padx='7', pady='7')
    enabled_label3.grid(row=row_count, column=1, padx='7', pady='7')

    row_count += 1
    txt1 = ttk.Label(frame_buttons, text="Шаг по времени", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry1 = ttk.Entry(frame_buttons)
    entry1.insert(0, dt)
    btn1 = Button(frame_buttons, text="Записать", command=show_message_1)
    label1 = ttk.Label(frame_buttons, text=f"{dt} секунд", padding=8)
    txt1.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry1.grid(row=row_count, column=1, padx='7', pady='7')
    btn1.grid(row=row_count, column=2, padx='7', pady='7')
    label1.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt2 = ttk.Label(frame_buttons, text="Количество аппаратов", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry2 = ttk.Entry(frame_buttons)
    entry2.insert(0, N_app)
    btn2 = Button(frame_buttons, text="Записать", command=show_message_2)
    label2 = ttk.Label(frame_buttons, text=f"{N_app} аппарат{okonchanye(N_app)}", padding=8)
    txt2.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry2.grid(row=row_count, column=1, padx='7', pady='7')
    btn2.grid(row=row_count, column=2, padx='7', pady='7')
    label2.grid(row=row_count, column=3, padx='7', pady='7')
    photo_app_1 = Image.open("icons/space2.png")
    photo_app_1 = photo_app_1.resize((50, 50))
    photo_plus = Image.open("icons/plus.png")
    photo_plus = photo_plus.resize((22, 22))
    photo_app_n = merge_n_photos(photo_app_1, photo_plus, N_app)
    img = ImageTk.PhotoImage(photo_app_n)
    b_app = Label(frame_buttons, image=img)
    b_app.grid(row=row_count, column=4, columnspan=4, sticky=W)

    row_count += 1
    txt3 = ttk.Label(frame_buttons, text="Сохранение раз в", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry3 = ttk.Entry(frame_buttons, state=entry3_state)
    entry3.insert(0, save_rate)
    btn3 = Button(frame_buttons, text="Записать", command=show_message_3)
    if entry3_state == 'NORMAL':
        label3 = ttk.Label(frame_buttons, text=f"{save_rate} итераций", padding=8)
    else:
        label3 = ttk.Label(frame_buttons, text=f"[изображения не сохраняются]", padding=8)
    txt3.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry3.grid(row=row_count, column=1, padx='7', pady='7')
    btn3.grid(row=row_count, column=2, padx='7', pady='7')
    label3.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    choice_1_vars = ['пробная\n(турист)', 'длинная\n(новичок)', 'антенна\n(мастер)', 'станция\n(Сэм)']
    choice_1 = StringVar(value=choice)  # по умолчанию будет выбран элемент с value=java
    label_choice_1 = ttk.Label(frame_buttons, text=f"Выбор конструкции:", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    label_choice_1_post = ttk.Label(frame_buttons, text=f"{choice_1_vars[int(choice)-1]}")
    choice_1_1 = ttk.Radiobutton(frame_buttons, text='1', value='1', variable=choice_1, command=choice_const)
    choice_1_2 = ttk.Radiobutton(frame_buttons, text='2', value='2', variable=choice_1, command=choice_const)
    choice_1_3 = ttk.Radiobutton(frame_buttons, text='3', value='3', variable=choice_1, command=choice_const)
    choice_1_4 = ttk.Radiobutton(frame_buttons, text='4', value='4', variable=choice_1, command=choice_const)
    photo_consts = [Image.open("icons/const_1.png"), Image.open("icons/const_2.png"),
                    Image.open("icons/const_3.png"), Image.open("icons/const_4.png")]
    size = (230, 150)
    photo_consts = [photo_consts[0].resize(size), photo_consts[1].resize(size),
                    photo_consts[2].resize(size), photo_consts[3].resize(size)]
    img1 = ImageTk.PhotoImage(photo_consts[int(choice)-1])
    b_const = Label(frame_buttons, image=img1)
    b_const.grid(row=row_count, column=3, rowspan=4)
    label_choice_1.grid(row=row_count, rowspan=4, column=0, padx='7', pady='7', sticky=EW)
    choice_1_1.grid(row=row_count, column=1, padx='7', pady='7')
    choice_1_2.grid(row=row_count+1, column=1, padx='7', pady='7')
    choice_1_3.grid(row=row_count+2, column=1, padx='7', pady='7')
    choice_1_4.grid(row=row_count+3, column=1, padx='7', pady='7')
    label_choice_1_post.grid(row=row_count, rowspan=4, column=2, padx='7', pady='7')

    row_count += 4
    choice_2_vars = ['без управления', 'импульсное', 'ПД-регулятор', 'ЛКР']
    choice_2 = StringVar(value=control)  # по умолчанию будет выбран элемент с value=java
    label_choice_2 = ttk.Label(frame_buttons, text=f"Выбор управления:", background="#9E9E9E", foreground="#E0EEE0",
                               padding=8)
    label_choice_2_post = ttk.Label(frame_buttons, text=f"{choice_2_vars[int(control) - 1]}")
    choice_2_1 = ttk.Radiobutton(frame_buttons, text='1', value='1', variable=choice_2, command=choice_control)
    choice_2_2 = ttk.Radiobutton(frame_buttons, text='2', value='2', variable=choice_2, command=choice_control)
    choice_2_3 = ttk.Radiobutton(frame_buttons, text='3', value='3', variable=choice_2, command=choice_control)
    choice_2_4 = ttk.Radiobutton(frame_buttons, text='4', value='4', variable=choice_2, command=choice_control)
    label_choice_2.grid(row=row_count, rowspan=4, column=0, padx='7', pady='7', sticky=EW)
    choice_2_1.grid(row=row_count, column=1, padx='7', pady='7')
    choice_2_2.grid(row=row_count+1, column=1, padx='7', pady='7')
    choice_2_3.grid(row=row_count+2, column=1, padx='7', pady='7')
    choice_2_4.grid(row=row_count+3, column=1, padx='7', pady='7')
    label_choice_2_post.grid(row=row_count, rowspan=4, column=3, padx='7', pady='7')

    row_count += 4
    labe1 = ttk.Label(frame_buttons, text="Параметры управления", background="#828282", foreground="#E0EEE0",
                      padding=8)
    labe1.grid(row=row_count, column=0, padx='7', pady='7', columnspan=4, sticky=EW)

    row_count += 1
    txt4 = ttk.Label(frame_buttons, text="Коэффициент ПД", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry4 = ttk.Entry(frame_buttons, state=entry4_state)
    entry4.insert(0, k_p)
    btn4 = Button(frame_buttons, text="Записать", command=show_message_4)
    if entry4_state == 'NORMAL':
        label4 = ttk.Label(frame_buttons, text=f"{k_p}", padding=8)
    else:
        label4 = ttk.Label(frame_buttons, text=f"[не управляется ПД-регулятором]", padding=8)
    txt4.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry4.grid(row=row_count, column=1, padx='7', pady='7')
    btn4.grid(row=row_count, column=2, padx='7', pady='7')
    label4.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt5 = ttk.Label(frame_buttons, text="Ускорение двигателя", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry5 = ttk.Entry(frame_buttons, state=entry5_state)
    entry5.insert(0, a_max)
    btn5 = Button(frame_buttons, text="Записать", command=show_message_5)
    if entry5_state == 'NORMAL':
        label5 = ttk.Label(frame_buttons, text=f"{int(a_max*1e3)} мН", padding=8)
    else:
        label5 = ttk.Label(frame_buttons, text=f"[нет непрерывного управления]", padding=8)
    txt5.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry5.grid(row=row_count, column=1, padx='7', pady='7')
    btn5.grid(row=row_count, column=2, padx='7', pady='7')
    label5.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt6 = ttk.Label(frame_buttons, text="Импульс двигателя", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry6 = ttk.Entry(frame_buttons, state=entry6_state)
    entry6.insert(0, du_impulse_max)
    btn6 = Button(frame_buttons, text="Записать", command=show_message_6)
    if entry6_state == 'NORMAL':
        label6 = ttk.Label(frame_buttons, text=f"{du_impulse_max} м/с", padding=8)
    else:
        label6 = ttk.Label(frame_buttons, text=f"[нет импульсного управления]", padding=8)
    txt6.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry6.grid(row=row_count, column=1, padx='7', pady='7')
    btn6.grid(row=row_count, column=2, padx='7', pady='7')
    label6.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    labe2 = ttk.Label(frame_buttons, text="Ограничения", background="#828282", foreground="#E0EEE0",
                      padding=8)
    labe2.grid(row=row_count, column=0, padx='7', pady='7', columnspan=4, sticky=EW)

    row_count += 1
    txt7 = ttk.Label(frame_buttons, text="Радиус опасной зоны", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry7 = ttk.Entry(frame_buttons)
    entry7.insert(0, d_crash)
    btn7 = Button(frame_buttons, text="Записать", command=show_message_7)
    label7 = ttk.Label(frame_buttons, text=f"{int(100*d_crash)} см", padding=8)
    txt7.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry7.grid(row=row_count, column=1, padx='7', pady='7')
    btn7.grid(row=row_count, column=2, padx='7', pady='7')
    label7.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt8 = ttk.Label(frame_buttons, text="Радиус захвата", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry8 = ttk.Entry(frame_buttons)
    entry8.insert(0, d_to_grab)
    btn8 = Button(frame_buttons, text="Записать", command=show_message_8)
    label8 = ttk.Label(frame_buttons, text=f"{int(d_to_grab*100)} см", padding=8)
    txt8.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry8.grid(row=row_count, column=1, padx='7', pady='7')
    btn8.grid(row=row_count, column=2, padx='7', pady='7')
    label8.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt9 = ttk.Label(frame_buttons, text="Время эпизода", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry9 = ttk.Entry(frame_buttons)
    entry9.insert(0, T_max)
    btn9 = Button(frame_buttons, text="Записать", command=show_message_9)
    label9 = ttk.Label(frame_buttons, text=f"{T_max} секунд, {int(100*T_max/(2*np.pi/o.w_hkw))}% оборота", padding=8)
    txt9.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry9.grid(row=row_count, column=1, padx='7', pady='7')
    btn9.grid(row=row_count, column=2, padx='7', pady='7')
    label9.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt10 = ttk.Label(frame_buttons, text="Скорость отталкивания", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry10 = ttk.Entry(frame_buttons)
    entry10.insert(0, u_max)
    btn10 = Button(frame_buttons, text="Записать", command=show_message_10)
    label10 = ttk.Label(frame_buttons, text=f"{u_max*10} см/с", padding=8)
    txt10.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry10.grid(row=row_count, column=1, padx='7', pady='7')
    btn10.grid(row=row_count, column=2, padx='7', pady='7')
    label10.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt11 = ttk.Label(frame_buttons, text="Угловая скорость", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry11 = ttk.Entry(frame_buttons)
    entry11.insert(0, w_max)
    btn11 = Button(frame_buttons, text="Записать", command=show_message_11)
    label11 = ttk.Label(frame_buttons, text=f"{w_max} рад/с, оборот раз в {2*np.pi/w_max} секунд", padding=8)
    txt11.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry11.grid(row=row_count, column=1, padx='7', pady='7')
    btn11.grid(row=row_count, column=2, padx='7', pady='7')
    label11.grid(row=row_count, column=3, padx='7', pady='7')

    row_count += 1
    txt12 = ttk.Label(frame_buttons, text="Отклонение станции", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry12 = ttk.Entry(frame_buttons)
    entry12.insert(0, j_max)
    btn12 = Button(frame_buttons, text="Записать", command=show_message_12)
    label12 = ttk.Label(frame_buttons, text=f"{j_max} градусов", padding=8)
    txt12.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry12.grid(row=row_count, column=1, padx='7', pady='7')
    btn12.grid(row=row_count, column=2, padx='7', pady='7')
    label12.grid(row=row_count, column=3, padx='7', pady='7')

    frame_buttons.update_idletasks()
    frame_canvas.config(width=1915,
                        height=920)
    canvas.config(scrollregion=canvas.bbox("all"))

    root.mainloop()
