from tkinter_files.assembly0 import *
from tkinter_files.tk_functions import *


def click_button_assembly0(event=None):
    from tkinter_files.assembly0 import click_button_assembly
    global root
    root.destroy()
    click_button_assembly()

def click_button_test0(event=None):
    from tkinter_files.testing0 import click_button_test
    global root
    root.destroy()
    click_button_test()

def click_button_talk(event=None):
    global root
    talk()

def click_button_plot(event=None):
    from tkinter_files.plotting0 import click_button_plot
    global root
    root.destroy()
    click_button_plot()

def click_button_info0(event=None):
    from tkinter_files.information import click_button_info
    global root
    root.destroy()
    click_button_info()

def return_home(event=None):
    global root
    root = Tk()
    icons = Icons()

    root.title("Проект Г.И.Б.О.Н.")
    root.geometry("1000x690+200+100")
    root.minsize(1000, 690)
    root.maxsize(1000, 690)
    root.iconphoto(True, icons.icon)

    img = ImageTk.PhotoImage(Image.open("icons/robot_talk1.png"))
    b = Label(image=img)
    b.grid(row=0, column=0, columnspan=5)

    def main_buttons(column: int, key: str, name: str, func: any, icon: any):
        btn = Button(text=name, command=func, image=icon, compound=LEFT)
        btn.grid(row=1, column=column, padx='7', pady='7')
        btn.bind("<Return>", func)
        root.bind(key, func)

    main_buttons(0, "q", "Поболтать", click_button_talk, icons.talk)
    main_buttons(1, "a", "Начать сборку", click_button_assembly0, icons.assembly)
    main_buttons(2, "t", "Тестировка", click_button_test0, icons.test)
    main_buttons(3, "g", "Графики", click_button_plot, icons.plot)
    main_buttons(4, "w", "Что я такое?", click_button_info0, icons.idea)
    root.focus_force()
    root.mainloop()


if __name__ == '__main__':
    return_home()
