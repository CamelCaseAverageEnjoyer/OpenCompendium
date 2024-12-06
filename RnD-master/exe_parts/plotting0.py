from NIR3 import return_home
from all_objects import *
from exe_parts.tk_functions import *
from exe_parts.plotting1 import *
from PIL import ImageTk
import subprocess
from tkinter import *
from tkinter import ttk
global w, dt, T_max, N, k_p
dt = 1.
w = 3e-3
T_max = 1000.
N = 5
k_p = 1e-4


def return_home0():
    from NIR3 import return_home
    global root
    root.destroy()
    return_home()


def animate_images():
    global entry_main
    params = entry_main.get().split()
    name = params[0]
    framerate = int(params[1])
    subprocess.Popen(["cd", "img"])
    subprocess.call(["ffmpeg", "-f", "image2", "-framerate", framerate, "-start_number", 1, "-i", "gibbon_%04d.png",
                    "-s", "1834x1051", "-b", "v", "10000k", f"../{name}.avi"])
    subprocess.call(["cd", ".."])


def rewrite_dt():
    global label1, entry1, dt
    dt = float(entry1.get())
    label1["text"] = f"{dt} секунд"


def rewrite_T():
    global label2, entry2, T_max
    T_max = float(entry2.get())
    label2["text"] = f"{T_max} секунд"


def rewrite_N():
    global label3, entry3, N
    N = int(entry3.get())
    label3["text"] = f"{N}"


def pd_control_params_search0():
    global label_t1, back_yes, back_run, dt, T_max, N, k_p
    label_t1["background"] = back_run
    label_t1["text"] = "Смотри график"
    k_p = pd_control_params_search(dt=dt, T_max=T_max, N=N)
    label_t1["background"] = back_yes
    label_t1["text"] = f"k={k_p}"


def plot_params_while_main0():
    global entry_t2
    plot_params_while_main(entry_t2.get())


def plot_a_avoid0():
    global entry_t3
    plot_a_avoid(entry_t3.get())


def copy_k_p():
    # import pyperclip
    import clipboard
    global k_p
    # pyperclip.copy(str(k_p))
    # pyperclip.paste()
    clipboard.copy(str(k_p))


def click_button_plot():
    global root, entry_main, dt, N
    global label1, entry1, label2, entry2, label3, entry3, entry_t2, entry_t3
    global label_t1, back_yes, back_run
    root = Tk()
    root.title("Проект Г.И.Б.О.Н.: показательные результаты")
    root.geometry("1980x1080+0+0")
    root.minsize(1000, 685)
    root.maxsize(1980, 1080)
    photo_home = PhotoImage(file="icons/home.png").subsample(10, 10)
    photo_operation = PhotoImage(file="icons/operation.png").subsample(10, 10)
    photo_processing = PhotoImage(file="icons/processing.png").subsample(10, 10)
    img_stat = PhotoImage(file="icons/statistics.png").subsample(10, 10)
    photo_save = PhotoImage(file="icons/save.png").subsample(10, 10)
    photo_anim = PhotoImage(file="icons/animation.png").subsample(10, 10)
    photo_down = PhotoImage(file="icons/download.png").subsample(10, 10)
    back_yes = "#1E90FF"
    back_no = "#8B5F65"
    back_run = "#483D8B"

    btn_home = Button(text="На главную", command=return_home0, image=photo_home, compound=LEFT)
    btn_main = Button(text="Анимировать результаты", command=animate_images, image=photo_anim, compound=LEFT)
    entry_main = EntryWithPlaceholder(root, '[Название] [Частота кадров]')
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=EW)
    btn_main.grid(row=0, column=1, padx='7', pady='7', sticky=EW)
    entry_main.grid(row=0, column=2, padx='7', pady='7', sticky=NSEW)

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
    labe2 = ttk.Label(frame_buttons, text="Делаем дела", background="#828282", foreground="#E0EEE0",
                      padding=8)
    labe2.grid(row=row_count, column=4, padx='7', pady='7', columnspan=4, sticky=EW)

    row_count += 1
    txt1 = ttk.Label(frame_buttons, text="Шаг по времени", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry1 = ttk.Entry(frame_buttons)
    entry1.insert(0, dt)
    btn1 = Button(frame_buttons, text="Записать", command=rewrite_dt)
    label1 = ttk.Label(frame_buttons, text=f"{dt} секунд", padding=8)
    txt1.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry1.grid(row=row_count, column=1, padx='7', pady='7')
    btn1.grid(row=row_count, column=2, padx='7', pady='7')
    label1.grid(row=row_count, column=3, padx='7', pady='7')

    btn_t1 = Button(frame_buttons, text="Подбор к-в ПД-регулятора", command=pd_control_params_search0, image=photo_operation, compound=LEFT)
    label_t1 = ttk.Label(frame_buttons, text="Не начато", background="#9E9E9E", foreground="#E0EEE0", padding=8, width=30)
    btn_t1_copy = Button(frame_buttons, text="Ctrl-C", command=copy_k_p)
    b_s1 = Label(frame_buttons, image=img_stat)
    btn_t1.grid(row=row_count, column=4, padx='7', pady='7', sticky=EW)
    label_t1.grid(row=row_count, column=5, padx='7', pady='7', sticky=NSEW)
    btn_t1_copy.grid(row=row_count, column=6, padx='7', pady='7', sticky=EW)
    b_s1.grid(row=row_count, column=7, sticky=W)

    row_count += 1
    txt2 = ttk.Label(frame_buttons, text="Время эпизода", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry2 = ttk.Entry(frame_buttons)
    entry2.insert(0, T_max)
    btn2 = Button(frame_buttons, text="Записать", command=rewrite_T)
    label2 = ttk.Label(frame_buttons, text=f"{T_max}", padding=8)
    txt2.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry2.grid(row=row_count, column=1, padx='7', pady='7')
    btn2.grid(row=row_count, column=2, padx='7', pady='7')
    label2.grid(row=row_count, column=3, padx='7', pady='7')

    btn_t2 = Button(frame_buttons, text="График движения аппаратов", command=plot_params_while_main0, image=photo_operation, compound=LEFT)
    entry_t2 = EntryWithPlaceholder(frame_buttons, '[Название]')
    b_s2 = Label(frame_buttons, image=img_stat)
    btn_t2.grid(row=row_count, column=4, padx='7', pady='7', sticky=EW)
    entry_t2.grid(row=row_count, column=5, padx='7', pady='7')
    b_s2.grid(row=row_count, column=7, sticky=W)

    row_count += 1
    txt3 = ttk.Label(frame_buttons, text="Число N", background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry3 = ttk.Entry(frame_buttons)
    entry3.insert(0, N)
    btn3 = Button(frame_buttons, text="Записать", command=rewrite_N)
    label3 = ttk.Label(frame_buttons, text=f"{N}", padding=8)
    txt3.grid(row=row_count, column=0, padx='7', pady='7', sticky=EW)
    entry3.grid(row=row_count, column=1, padx='7', pady='7')
    btn3.grid(row=row_count, column=2, padx='7', pady='7')
    label3.grid(row=row_count, column=3, padx='7', pady='7')

    btn_t3 = Button(frame_buttons, text="Эпюра огибающих ускорений", command=plot_a_avoid0, image=photo_operation, compound=LEFT)
    entry_t3 = EntryWithPlaceholder(frame_buttons, '[Название]')
    b_s3 = Label(frame_buttons, image=img_stat)
    btn_t3.grid(row=row_count, column=4, padx='7', pady='7', sticky=EW)
    entry_t3.grid(row=row_count, column=5, padx='7', pady='7')
    b_s3.grid(row=row_count, column=7, sticky=W)

    ############################################################################################
    frame_buttons.update_idletasks()
    frame_canvas.config(width=1915,
                        height=920)
    canvas.config(scrollregion=canvas.bbox("all"))

    root.mainloop()

