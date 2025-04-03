import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import scipy
import math

def get_vars(name: str, n: int):
    """Генерит символьные переменные с одинаковым названием и индексами 0...(n-1)
    :param name: Название переменных (без учёта индекса)
    :param n: Количество переменных"""
    s = ""
    for i in range(n):
        s += f"{name}_{i} "
    return var(s, real=True)

def my_print(txt: any, color: str = None, if_print: bool = True, bold: bool = False, if_return: bool = False,
             end: str = '\n') -> None:
    import colorama
    """Функция вывода цветного текста
    :param txt: Выводимый текст
    :param color: Цвет текста {b, g, y, r, c, m}
    :param if_print: Флаг вывода для экономии места
    :param bold: Жирный текст
    :param if_return: Надо ли возвращать строку"""
    color_bar = {"b": colorama.Fore.BLUE, "g": colorama.Fore.GREEN, "y": colorama.Fore.YELLOW, "r": colorama.Fore.RED,
                 "c": colorama.Fore.CYAN, "m": colorama.Fore.MAGENTA, None: colorama.Style.RESET_ALL}
    _txt = f"\033[1m{txt}\033[0m" if bold else txt
    anw = color_bar[color] + f"{_txt}" + colorama.Style.RESET_ALL
    if if_print and color in color_bar.keys():
        print(anw, end=end)
    if if_return:
        return anw