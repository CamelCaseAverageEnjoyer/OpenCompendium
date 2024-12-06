import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
from all_objects import *


class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder=None):
        self.entry_var = tk.StringVar()
        super().__init__(master, textvariable=self.entry_var)

        if placeholder is not None:
            self.placeholder = placeholder
            self.placeholder_color = 'grey'
            self.default_fg_color = self['fg']
            self.placeholder_on = False
            self.put_placeholder()

            self.entry_var.trace("w", self.entry_change)

            # При всех перечисленных событиях, если placeholder отображается, ставить курсор на 0 позицию
            self.bind("<FocusIn>", self.reset_cursor)
            self.bind("<KeyRelease>", self.reset_cursor)
            self.bind("<ButtonRelease>", self.reset_cursor)

    def entry_change(self, *args):
        if not self.get():
            self.put_placeholder()
        elif self.placeholder_on:
            self.remove_placeholder()
            self.entry_change()  # На случай, если после удаления placeholder остается пустое поле

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color
        self.icursor(0)
        self.placeholder_on = True

    def remove_placeholder(self):
        # Если был вставлен какой-то символ в начало, удаляем не весь текст, а только placeholder:
        text = self.get()[:-len(self.placeholder)]
        self.delete('0', 'end')
        self['fg'] = self.default_fg_color
        self.insert(0, text)
        self.placeholder_on = False

    def reset_cursor(self, *args):
        if self.placeholder_on:
            self.icursor(0)

class Icons:
    def __init__(self):
        self.talk = PhotoImage(file="icons/discussing.png").subsample(10, 10)
        self.test = PhotoImage(file="icons/processing.png").subsample(10, 10)
        self.plot = PhotoImage(file="icons/vision.png").subsample(10, 10)
        self.home = PhotoImage(file="icons/home.png").subsample(10, 10)
        self.assembly = PhotoImage(file="icons/solution.png").subsample(10, 10)
        self.idea = PhotoImage(file="icons/idea.png").subsample(10, 10)
        self.icon = PhotoImage(file="icons/satellite.png").subsample(10, 10)
        self.what = PhotoImage(file="icons/what.png").subsample(10, 10)
        self.save = PhotoImage(file="icons/save.png").subsample(10, 10)
        self.down = PhotoImage(file="icons/download.png").subsample(10, 10)
        self.next = PhotoImage(file="icons/next.png").subsample(10, 10)
        self.stat = PhotoImage(file="icons/statistics.png").subsample(10, 10)
        self.oper = PhotoImage(file="icons/operation.png").subsample(10, 10)
        self.proc = PhotoImage(file="icons/processing.png").subsample(10, 10)
        self.anim = PhotoImage(file="icons/animation.png").subsample(10, 10)
        self.app_1 = Image.open("icons/space2.png")
        self.app_1 = self.app_1.resize((50, 50))
        self.plus = Image.open("icons/plus.png")
        self.plus = self.plus.resize((22, 22))
        self.back = PhotoImage(file="icons/back.png").subsample(10, 10)
        self.what = PhotoImage(file="icons/what.png").subsample(10, 10)
        self.back_yes = "#1E90FF"
        self.back_no = "#8B5F65"
        self.back_run = "#483D8B"

def create_check(name: str, default_value: bool, n_row: int, cmd, frame):
    """Добавление флажка на окно                                                                \n
    Используется в структуре grid -> необходимо подать row в структуре                          \n
    Используйте в виде:                                                                         \n
    row, check_var_n, checkbutton_n, check_label_n = create_check(name, default_value, row, cmd)"""
    check_var = IntVar()
    check_var.set(default_value)
    checkbutton = ttk.Checkbutton(frame, text=name, variable=check_var, command=cmd)
    check_label = ttk.Label(frame, text=default_value)
    checkbutton.grid(row=n_row, column=0, padx='7', pady='7', sticky=E)
    check_label.grid(row=n_row, column=1, padx='7', pady='7')
    return n_row + 1, check_var, checkbutton, check_label

def create_label(name: str, n_row: int, frame, width: int = 4):
    label = ttk.Label(frame, text=name, background="#828282", foreground="#E0EEE0", padding=8)
    label.grid(row=n_row, column=0, padx='7', pady='7', columnspan=width, sticky=EW)
    return n_row + 1, label

def create_entry(name: str, default_value: [float, int], n_row: int, cmd, frame):
    txt = ttk.Label(frame, text=name, background="#9E9E9E", foreground="#E0EEE0", padding=8)
    entry = ttk.Entry(frame)
    entry.insert(0, default_value)
    btn = Button(frame, text="Записать", command=cmd)
    label = ttk.Label(frame, text=f"{default_value}", padding=8)
    txt.grid(row=n_row, column=0, padx='7', pady='7', sticky=EW)
    entry.grid(row=n_row, column=1, padx='7', pady='7')
    btn.grid(row=n_row, column=2, padx='7', pady='7')
    label.grid(row=n_row, column=3, padx='7', pady='7')
    btn.bind("<Return>", cmd)
    return n_row + 1, label, entry

def create_choice(name: str, default_value: str, n_row: int, n_vars: int, cmd, frame, size=(230, 150)):
    choice = StringVar(value=default_value)
    label_choice = ttk.Label(frame, text=name, background="#9E9E9E", foreground="#E0EEE0", padding=8)
    label_choice_extra = ttk.Label(frame, text="здесь должен быть текст")
    choice_n = []
    for j in range(n_vars):
        choice_n.append(ttk.Radiobutton(frame, text=str(j+1), value=str(j+1), variable=choice, command=cmd))
        choice_n[j].grid(row=n_row + j, column=1, padx='7', pady='7')
    photo = Image.open("icons/question.png").resize(size)
    img = ImageTk.PhotoImage(photo)
    img_label = Label(frame, image=img)
    img_label.grid(row=n_row, column=3, rowspan=4)
    label_choice.grid(row=n_row, rowspan=4, column=0, padx='7', pady='7', sticky=EW)
    label_choice_extra.grid(row=n_row, rowspan=4, column=2, padx='7', pady='7')
    return n_row + n_vars, choice, label_choice, label_choice_extra, img_label


def get_simple_label(name: str, frame):
    return ttk.Label(frame, text=name, background="#828282", foreground="#E0EEE0", padding=8, width=20)

def local_label(name: str, n_row: int, n_col: int, column_span: int, frame: any):
    label = ttk.Label(frame, text=name, background="#828282", foreground="#E0EEE0", padding=8)
    label.grid(row=n_row, column=n_col, padx='7', pady='7', columnspan=column_span, sticky=EW)
