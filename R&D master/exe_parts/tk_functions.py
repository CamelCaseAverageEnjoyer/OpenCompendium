import tkinter as tk
from tkinter import ttk


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
