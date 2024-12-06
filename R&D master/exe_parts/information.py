from NIR3 import return_home
from all_objects import *
from test import *
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


def return_home0():
    from NIR3 import return_home
    global root
    root.destroy()
    return_home()


def click_button_info():
    global root
    root = Tk()
    root.title("Проект Г.И.Б.О.Н.: информация")
    root.geometry("1980x1080+0+0")
    root.minsize(1000, 685)
    root.maxsize(1980, 1080)
    photo_home = PhotoImage(file="icons/home.png").subsample(10, 10)
    back_yes = "#1E90FF"
    back_no = "#8B5F65"

    btn_home = Button(text="На главную", command=return_home0, image=photo_home, compound=LEFT)
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=W)

    img = ImageTk.PhotoImage(Image.open("icons/info.png"))
    b = Label(image=img)
    b.grid(row=1, column=0)

    T = f"Очень хорошо. Мы двое — прекрасная команда… благодаря тому из нас, кто делает всю аботу. Иначе говоря, во " \
        f"мне есть 3 основных отдела:\n1.   Сборка\n        Задайте параметры и начните расчёт! Обратите внимание, вы" \
        f"можете сохранить параметры и использовать их в следующий раз!\n"

    # label_1 = ttk.Label(text=T)
    # label_1.grid(row=1, column=0, padx='7', pady='7', sticky=EW)

    root.mainloop()
