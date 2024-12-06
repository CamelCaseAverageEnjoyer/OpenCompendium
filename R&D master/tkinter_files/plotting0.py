from tkinter_files.tk_functions import *
from mylibs.result_show_functions import *
import subprocess
params = {'dt': 10., 'w': 3e-3, 'T_max': 500., 'N': 2, 'k_p': 1e-4}


def return_home0(event=None):
    from main_interface import return_home
    global root
    root.destroy()
    return_home()

def animate_images():
    global entry_main
    params_local = entry_main.get().split()
    name = params_local[0]
    framerate = int(params_local[1])
    subprocess.Popen(["cd", "img"])
    subprocess.call(["ffmpeg", "-f", "image2", "-framerate", framerate, "-start_number", 1, "-i", "gibbon_%04d.png",
                    "-s", "1834x1051", "-b", "v", "10000k", f"../{name}.avi"])
    subprocess.call(["cd", ".."])

def rewrite_dt(event=None):
    global label_1, entry_1, params
    params['dt'] = float(entry_1.get())
    label_1["text"] = f"{params['dt']} секунд"

def rewrite_t(event=None):
    global label_2, entry_2, params
    params['T_max'] = float(entry_2.get())
    label_2["text"] = f"{params['T_max']} секунд"

def rewrite_n(event=None):
    global label_3, entry_3, params
    params['N'] = int(entry_3.get())
    label_3["text"] = f"{params['N']}"

def rewrite_w(event=None):
    global label_4, entry_4, params
    params['w'] = float(entry_4.get())
    label_4["text"] = f"{params['w']}"

def pd_control_params_search0():
    global label_t1, params, icons
    label_t1["background"] = icons.back_run
    label_t1["text"] = "Смотри график"
    params['k_p'] = pd_control_params_search(dt=params['dt'], T_max=params['T_max'], n_p=params['N'])
    label_t1["background"] = icons.back_yes
    label_t1["text"] = f"k={params['k_p']}"

def plot_avoid_field_params():
    global label_t4, params, icons
    label_t4["background"] = icons.back_run
    label_t4["text"] = "Смотри график"
    plot_avoid_field_params_search(dt=params['dt'], T_max=params['T_max'], N=params['N'])
    label_t4["background"] = icons.back_yes
    label_t4["text"] = f"Посчитано"

def copy_k_p():
    import clipboard
    global params
    clipboard.copy(str(params['k_p']))


def click_button_plot():
    global root, entry_main, params, icons
    global label_1, entry_1, label_2, entry_2, label_3, entry_3, label_4, entry_4
    global label_t1, label_t4, label_t5
    root = Tk()
    icons = Icons()

    root.title("Проект Г.И.Б.О.Н.: показательные результаты")
    root.geometry("1980x1080+0+0")
    root.minsize(1000, 685)
    root.maxsize(1980, 1080)

    btn_home = Button(text="На главную", command=return_home0, image=icons.home, compound=LEFT)
    btn_main = Button(text="Анимировать результаты", command=animate_images, image=icons.anim, compound=LEFT)
    entry_main = EntryWithPlaceholder(root, '[Название] [Частота кадров]')
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=EW)
    btn_main.grid(row=0, column=1, padx='7', pady='7', sticky=EW)
    entry_main.grid(row=0, column=2, padx='7', pady='7', sticky=NSEW)
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

    def create_result_show_button(name: str, cmd: any, extra: str = None,
                                  btn_2_name: list = None, btn_2_cmd: list = None):
        button = Button(frame, text=name, command=cmd, image=icons.oper, compound=LEFT)
        button.grid(row=row, column=4, padx='7', pady='7', sticky=EW)
        b_stat = Label(frame, image=icons.stat)
        b_stat.grid(row=row, column=8, sticky=W)
        if btn_2_name is not None:
            for i in range(len(btn_2_name)):
                btn_2 = Button(frame, text=btn_2_name[i], command=btn_2_cmd[i])
                btn_2.grid(row=row, column=6 + i, padx='7', pady='7', sticky=EW)
        if extra == 'label':
            label = ttk.Label(frame, text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8, width=30)
            label.grid(row=row, column=5, padx='7', pady='7', sticky=NSEW)
            return label
        elif extra == 'entry':
            entry = EntryWithPlaceholder(frame, '[Название]')
            entry.grid(row=row, column=5, padx='7', pady='7')
            return entry

    row = 0
    local_label("Параметры управления", row, 0, 4, frame)
    local_label("Делаем дела", row, 4, 5, frame)
    row += 1

    label_t1 = create_result_show_button("Подбор к-в ПД-регулятора", pd_control_params_search0, 'label',
                                         ["Ctrl-C", "Show"], [copy_k_p, reader_pd_control_params])
    row, label_1, entry_1 = create_entry("Шаг по времени", params['dt'], row, rewrite_dt, frame)
    rewrite_dt()

    create_result_show_button("График движения аппаратов", plot_params_while_main)
    row, label_2, entry_2 = create_entry("Время эпизода", params['T_max'], row, rewrite_t, frame)
    rewrite_t()

    create_result_show_button("Эпюра огибающих ускорений", plot_a_avoid)
    row, label_3, entry_3 = create_entry("Число N", params['N'], row, rewrite_n, frame)
    rewrite_n()

    label_t4 = create_result_show_button("Эпюра огибающих ускорений", plot_avoid_field_params, 'label',
                                         ["Show"], [reader_avoid_field_params_search])
    row, label_4, entry_4 = create_entry("Угловая скорость", params['w'], row, rewrite_w, frame)
    rewrite_w()

    create_result_show_button("Что ты такое чел", plot_repulsion_error) 
    row += 1

    label_t5 = create_result_show_button("Затраты скорости", dv_col_noncol_difference, 'label',
                                         ["Show"], [reader_dv_col_noncol_difference])
    row += 1

    frame.update_idletasks()
    frame_canvas.config(width=1915, height=920)
    canvas.config(scrollregion=canvas.bbox("all"))
    root.focus_force()
    root.mainloop()

