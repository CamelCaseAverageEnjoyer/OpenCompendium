from tkinter_files.tk_functions import *


def return_home0(event=None):
    from main_interface import return_home
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

    btn_home = Button(text="На главную", command=return_home0, image=photo_home, compound=LEFT)
    btn_home.grid(row=0, column=0, padx='7', pady='7', sticky=W)
    btn_home.bind("<Return>", return_home0)
    root.bind("h", return_home0)

    img = ImageTk.PhotoImage(Image.open("icons/info.png"))
    b = Label(image=img)
    b.grid(row=1, column=0, columnspan=4)

    T = f"Очень хорошо. Мы двое — прекрасная команда… благодаря тому из нас, кто делает всю аботу! \n\n" \
        f"Во мне есть 3 основных отдела:\n1.   Сборка\n        Задайте параметры и начните расчёт! Обратите внимание," \
        f" вы можете сохранить параметры и использовать их в следующий раз!\n2.   Тестировка\n" \
        f"        Мини-эксперименты, которые не войдут в статейку\n3.   Графики\n" \
        f"        Мини-эксперименты, которые войдёт в статейку))))\n\n" \
        f"Отлично, творец будущего! Однако, если вы стары, недоразвиты или страдаете от увечий или радиации в \n" \
        f"такой степени, что будущее не должно начинаться с вас, вернитесь в своё примитивное племя и пришлите\n" \
        f"кого-нибудь более подходящего для тестов."

    label = ttk.Label(text=T)
    label.grid(row=1, column=0)

    root.focus_force()
    root.mainloop()
