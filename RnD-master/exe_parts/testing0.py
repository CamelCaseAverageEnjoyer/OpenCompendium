from NIR3 import return_home
from all_objects import *
from mylibs.test_functions import *
from PIL import ImageTk
from test import *
from tkinter import *
from tkinter import ttk
global ORDER, w, dt, T_max
ORDER = 5
dt = 1.
w = 3e-3
T_max = 1000.


def return_home0():
    from NIR3 import return_home
    global root
    root.destroy()
    return_home()


def def_o():
    o1 = AllProblemObjects(if_impulse_control=True, if_PID_control=False,
                           is_saving=False,  diff_evolve_vectors=20, dt=0.5, if_talk=True)
    o1.w = np.array([0, -3e-5, 0])
    o1.Om = o1.w + o1.w_hkw_vec
    return o1


def full_test():
    global label_full, back_yes, back_no, ORDER, w, dt, T_max
    res = test_full_energy(order=ORDER, dt=dt, w=w)
    label_full["background"] = back_yes if res else back_no
    label_full["text"] = "Всё хорошо" if res else "Ничего хорошего"


def test_energy():
    global label_t1, back_t1, back_no, back_yes, back_run, ORDER, w, dt, T_max
    label_t1["background"] = back_run
    label_t1["text"] = "Смотри график"
    res = test_full_energy(order=ORDER, dt=dt, w=w, T_max=T_max)
    label_t1["background"] = back_yes if res else back_no
    label_t1["text"] = "Всё хорошо" if res else "Ничего хорошего"


def test_matrix():
    global label_t2, back_t2, back_no, back_yes, back_run, ORDER, w, dt, T_max
    res = test_rotation(order=ORDER, dt=dt, w=w, T_max=T_max)
    label_t2["background"] = back_yes if res else back_no
    label_t2["text"] = "Всё хорошо" if res else "Ничего хорошего"


def test_rk():
    global label_t3, back_t3, back_no, back_yes, back_run, ORDER, w, dt, T_max
    res = test_runge_kutta(order=ORDER, dt=dt, w=w, T_max=T_max)
    label_t3["background"] = back_yes if res else back_no
    label_t3["text"] = "Всё хорошо" if res else "Ничего хорошего"


def rewrite_order():
    global label1, entry1, ORDER
    ORDER = int(float(entry1.get()))
    label1["text"] = f"Требуемый порядок {ORDER}"


def rewrite_dt():
    global label2, entry2, dt
    dt = float(entry2.get())
    label2["text"] = f"{dt} секунд"


def rewrite_w():
    global label3, entry3, w, o
    w = float(entry3.get())
    label3["text"] = f"{w} рад/с ({'%.2f' % (100*w / o.w_hkw)}%)"


def rewrite_T():
    global label4, entry4, T_max, o
    T_max = float(entry4.get())
    label4["text"] = f"{T_max} секунд, {int(100*T_max/(2*np.pi/o.w_hkw))}% оборота"


def save_params():
    global root


def download_params():
    global root


def click_button_test():
    global root, label_full, back_yes, back_no, back_run, ORDER, w, T_max
    global label1, entry1, label2, entry2, label3, entry3, dt, label_t1, back_t1, label4, entry4
    global label_t2, back_t2, label_t3, back_t3, o
    root = Tk()
    root.title("Проект Г.И.Б.О.Н.: тестирование")
    root.geometry("1980x1080+0+0")
    root.minsize(1000, 685)
    root.maxsize(1980, 1080)
    photo_home = PhotoImage(file="icons/home.png").subsample(10, 10)
    photo_operation = PhotoImage(file="icons/operation.png").subsample(10, 10)
    photo_processing = PhotoImage(file="icons/processing.png").subsample(10, 10)
    img_stat = PhotoImage(file="icons/statistics.png").subsample(10, 10)
    photo_save = PhotoImage(file="icons/save.png").subsample(10, 10)
    photo_down = PhotoImage(file="icons/download.png").subsample(10, 10)
    back_yes = "#1E90FF"
    back_no = "#8B5F65"
    back_run = "#483D8B"

    btn_home = Button(text="На главную", command=return_home0, image=photo_home, compound=LEFT)
    btn_full = Button(text="Полное тестирование", command=full_test, image=photo_processing, compound=LEFT)
    label_full = ttk.Label(text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=EW)
    btn_full.grid(row=0, column=1, padx='7', pady='7', sticky=EW)
    label_full.grid(row=0, column=2, padx='7', pady='7', sticky=NSEW)

    o = AllProblemObjects()
    ############################################################################################
    frame_canvas = Frame(root)
    frame_canvas.grid(row=1, column=0, columnspan=3, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    frame_canvas.grid_propagate(False)
    canvas = Canvas(frame_canvas)  # , bg="yellow")
    canvas.grid(row=0, column=0, sticky="news")
    # vsb = Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    # vsb.grid(row=0, column=1, sticky='ns')
    # vsb2 = Scrollbar(frame_canvas, orient="horizontal", command=canvas.xview)
    # vsb2.grid(row=1, column=0, sticky='ew')
    canvas.configure()  # xscrollcommand=vsb2.set)  # , yscrollcommand=vsb.set)
    frame_buttons = Frame(canvas)  # , bg="blue")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')
    ############################################################################################
    row_count = 0
    labe1 = ttk.Label(frame_buttons, text="Параметры управления", background="#828282", foreground="#E0EEE0",
                      padding=8)
    labe1.grid(row=row_count, column=0, padx='7', pady='7', columnspan=4, sticky=EW)
    labe2 = ttk.Label(frame_buttons, text="Отделённые тесты", background="#828282", foreground="#E0EEE0",
                      padding=8)
    labe2.grid(row=row_count, column=4, padx='7', pady='7', columnspan=3, sticky=EW)
    labe3 = ttk.Label(frame_buttons, text="Сохранить параметры", background="#828282", foreground="#E0EEE0",
                      padding=8)
    labe3.grid(row=row_count, column=7, padx='7', pady='7', columnspan=2, sticky=EW)

    row_count += 1
    txt1 = ttk.Label(frame_buttons, text="Порядок точности", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry1 = ttk.Entry(frame_buttons)
    entry1.insert(0, ORDER)
    btn1 = Button(frame_buttons, text="Записать", command=rewrite_order)
    label1 = ttk.Label(frame_buttons, text=f"Требуемый порядок {ORDER}", padding=8, width=30)
    txt1.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry1.grid(row=row_count, column=1, padx='7', pady='7')
    btn1.grid(row=row_count, column=2, padx='7', pady='7')
    label1.grid(row=row_count, column=3, padx='7', pady='7', sticky=W)

    btn_t1 = Button(frame_buttons, text="Сохранение энергии", command=test_energy, image=photo_operation, compound=LEFT)
    label_t1 = ttk.Label(frame_buttons, text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8, width=30)
    b_s1 = Label(frame_buttons, image=img_stat)
    btn_t1.grid(row=row_count, column=4, padx='7', pady='7', sticky=EW)
    label_t1.grid(row=row_count, column=5, padx='7', pady='7', sticky=NSEW)
    b_s1.grid(row=row_count, column=6, sticky=W)

    btn_save = Button(frame_buttons, text="Сохранить", command=save_params, image=photo_save, compound=LEFT)
    btn_down = Button(frame_buttons, text="Загрузить", command=download_params, image=photo_down, compound=LEFT)
    btn_save.grid(row=row_count, column=7, padx='7', pady='7', sticky=EW)
    btn_down.grid(row=row_count, column=8, padx='7', pady='7', sticky=EW)

    row_count += 1
    txt2 = ttk.Label(frame_buttons, text="Шаг по времени", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry2 = ttk.Entry(frame_buttons)
    entry2.insert(0, dt)
    btn2 = Button(frame_buttons, text="Записать", command=rewrite_dt)
    label2 = ttk.Label(frame_buttons, text=f"{dt} секунд", padding=8)
    txt2.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry2.grid(row=row_count, column=1, padx='7', pady='7')
    btn2.grid(row=row_count, column=2, padx='7', pady='7')
    label2.grid(row=row_count, column=3, padx='7', pady='7', sticky=W)

    btn_t2 = Button(frame_buttons, text="Переходы систем координат", command=test_matrix, image=photo_operation, compound=LEFT)
    label_t2 = ttk.Label(frame_buttons, text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    btn_t2.grid(row=row_count, column=4, padx='7', pady='7', sticky=EW)
    label_t2.grid(row=row_count, column=5, padx='7', pady='7', sticky=NSEW)

    row_count += 1
    txt3 = ttk.Label(frame_buttons, text="Начальное вращение", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry3 = ttk.Entry(frame_buttons)
    entry3.insert(0, w)
    btn3 = Button(frame_buttons, text="Записать", command=rewrite_w)
    label3 = ttk.Label(frame_buttons, text=f"{w} рад/с ({'%.2f' % (100*w / o.w_hkw)}%)", padding=8)
    txt3.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry3.grid(row=row_count, column=1, padx='7', pady='7')
    btn3.grid(row=row_count, column=2, padx='7', pady='7')
    label3.grid(row=row_count, column=3, padx='7', pady='7', sticky=W)

    btn_t3 = Button(frame_buttons, text="Рунге-Кутта", command=test_rk, image=photo_operation, compound=LEFT)
    label_t3 = ttk.Label(frame_buttons, text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    btn_t3.grid(row=row_count, column=4, padx='7', pady='7', sticky=EW)
    label_t3.grid(row=row_count, column=5, padx='7', pady='7', sticky=NSEW)

    row_count += 1
    txt4 = ttk.Label(frame_buttons, text="Время эпизода", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry4 = ttk.Entry(frame_buttons)
    entry4.insert(0, T_max)
    btn4 = Button(frame_buttons, text="Записать", command=rewrite_T)
    label4 = ttk.Label(frame_buttons, text=f"{T_max} секунд, {int(100*T_max/(2*np.pi/o.w_hkw))}% оборота", padding=8)
    txt4.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry4.grid(row=row_count, column=1, padx='7', pady='7')
    btn4.grid(row=row_count, column=2, padx='7', pady='7')
    label4.grid(row=row_count, column=3, padx='7', pady='7', sticky=W)

    ############################################################################################
    frame_buttons.update_idletasks()
    frame_canvas.config(width=1915,
                        height=920)
    canvas.config(scrollregion=canvas.bbox("all"))

    root.mainloop()
