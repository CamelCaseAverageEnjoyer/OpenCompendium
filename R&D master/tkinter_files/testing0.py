from mylibs.test_functions import *
from tkinter_files.tk_functions import *
params = {'order': 2, 'dt': 1., 'w': 3e-7, 'T_max': 5000., 'N': 10}
o = AllProblemObjects()


def return_home0(event=None):
    from main_interface import return_home
    global root
    root.destroy()
    return_home()


def def_o():
    o_local = AllProblemObjects(if_impulse_control=True, if_PID_control=False,
                                is_saving=False,  diff_evolve_vectors=20, dt=0.5, if_talk=True)
    o_local.w = np.array([0, -3e-5, 0])
    o_local.Om = o_local.w + o_local.w_hkw_vec
    return o_local


def test_energy():
    global label_t1, back_t1, params, icons
    label_t1["background"] = icons.back_run
    label_t1["text"] = "Смотри график"
    res = test_full_energy(order=params['order'], dt=params['dt'], w=params['w'], T_max=params['T_max'])
    label_t1["background"] = icons.back_yes if res else icons.back_no
    label_t1["text"] = "Всё хорошо" if res else "Ничего хорошего"


def test_matrix():
    global label_t2, back_t2, icons, params
    res = test_rotation(order=params['order'], dt=params['dt'], w=params['w'], T_max=params['T_max'])
    label_t2["background"] = icons.back_yes if res else icons.back_no
    label_t2["text"] = "Всё хорошо" if res else "Ничего хорошего"


def test_rk():
    global label_t3, icons
    res = test_runge_kutta(order=params['order'], dt=params['dt'], w=params['w'], T_max=params['T_max'])
    label_t3["background"] = icons.back_yes if res else icons.back_no
    label_t3["text"] = "Всё хорошо" if res else "Ничего хорошего"

def test_collision():
    global params
    test_collision_map(n=params['N'])

def rewrite_order():
    global label_1, entry_1
    params['order'] = int(float(entry_1.get()))
    label_1["text"] = f"Требуемый порядок {params['order']}"


def rewrite_dt():
    global label_2, entry_2
    params['dt'] = float(entry_2.get())
    label_2["text"] = f"{params['dt']} секунд"


def rewrite_w():
    global label_3, entry_3, o
    params['w'] = float(entry_3.get())
    label_3["text"] = f"{params['w']} рад/с ({'%.2f' % (100 * params['w'] / o.w_hkw)}%)"


def rewrite_t():
    global label_4, entry_4, o
    params['T_max'] = float(entry_4.get())
    label_4["text"] = f"{params['T_max']} секунд, {int(100 * params['T_max'] / (2 * np.pi / o.w_hkw))}% оборота"


def rewrite_n():
    global label_5, entry_5
    params['N'] = int(entry_5.get())
    label_5["text"] = f"{params['N']}"


def save_params():
    global root


def download_params():
    global root


def full_test():
    test_energy()
    test_matrix()
    test_rk()

def click_button_test():
    global root, label_full, icons, params
    global label_1, label_2, label_3, label_t1, back_t1, label_4, label_5
    global entry_1, entry_2, entry_3, entry_4, entry_5
    global label_t2, back_t2, label_t3, back_t3
    root = Tk()
    icons = Icons()

    root.title("Проект Г.И.Б.О.Н.: тестирование")
    root.geometry("1980x1080+0+0")
    root.minsize(1000, 685)
    root.maxsize(1980, 1080)

    btn_home = Button(text="На главную", command=return_home0, image=icons.home, compound=LEFT)
    btn_full = Button(text="Полное тестирование", command=full_test, image=icons.proc, compound=LEFT)
    label_full = ttk.Label(text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=EW)
    btn_full.grid(row=0, column=1, padx='7', pady='7', sticky=EW)
    label_full.grid(row=0, column=2, padx='7', pady='7', sticky=NSEW)
    root.bind("h", return_home0)

    frame_canvas = Frame(root)
    frame_canvas.grid(row=1, column=0, columnspan=3, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    frame_canvas.grid_propagate(False)
    canvas = Canvas(frame_canvas)
    canvas.grid(row=0, column=0, sticky="news")
    canvas.configure()

    frame = Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor='nw')
    row = 0

    def local_test(name: str, n_row: int, n_col: int, func: any, frame: any, plot: bool = False):
        button = Button(frame, text=name, command=func, image=icons.oper, compound=LEFT)
        label = ttk.Label(frame, text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8, width=30)
        button.grid(row=n_row, column=n_col, padx='7', pady='7', sticky=EW)
        label.grid(row=n_row, column=n_col + 1, padx='7', pady='7', sticky=NSEW)
        if plot:
            label_stat = Label(frame, image=icons.stat)
            label_stat.grid(row=n_row, column=n_col + 2, sticky=W)
        return label

    local_label("Параметры управления", row, 0, 4, frame)
    local_label("Отделённые тесты", row, 4, 3, frame)
    local_label("Сохранить параметры", row, 7, 2, frame)
    row += 1

    btn_save = Button(frame, text="Сохранить", command=save_params, image=icons.save, compound=LEFT)
    btn_down = Button(frame, text="Загрузить", command=download_params, image=icons.down, compound=LEFT)
    btn_save.grid(row=row, column=7, padx='7', pady='7', sticky=EW)
    btn_down.grid(row=row, column=8, padx='7', pady='7', sticky=EW)

    label_t1 = local_test("Сохранение энергии", row, 4, test_energy, frame, True)
    row, label_1, entry_1 = create_entry("Порядок точности", params['order'], row, rewrite_order, frame)
    rewrite_order()

    label_t2 = local_test("Переходы систем координат", row, 4, test_matrix, frame, True)
    row, label_2, entry_2 = create_entry("Шаг по времени", params['dt'], row, rewrite_dt, frame)
    rewrite_dt()

    label_t3 = local_test("Рунге-Кутта", row, 4, test_rk, frame, True)
    row, label_3, entry_3 = create_entry("Начальное вращение", params['w'], row, rewrite_w, frame)
    rewrite_w()

    label_t4 = local_test("Столкновение", row, 4, test_collision, frame, True)
    row, label_4, entry_4 = create_entry("Время эпизода", params['T_max'], row, rewrite_t, frame)
    rewrite_t()

    row, label_5, entry_5 = create_entry("Число N", params['N'], row, rewrite_n, frame)
    rewrite_n()

    frame.update_idletasks()
    frame_canvas.config(width=1915, height=920)
    canvas.config(scrollregion=canvas.bbox("all"))
    root.focus_force()
    root.mainloop()
