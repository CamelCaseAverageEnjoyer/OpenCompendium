"""
Файл сгененрирован программой OpenCompendium/RnD-PhD/notebooks/dynamic.ipynb (Раздел: Генерация файла H.matrix.py)
Копия файла из каталога OpenCompendium/RnD-PhD/notebooks/dynamic.ipynb     ycharmProjects/PythonCompendium/DissertationPhd/storage/observability_mapping_partial_derivatives.py
"""
import numpy as np
from flexmath import *

def h_element(i_x, i_y, i_n, i, j, gm_1: str, gm_2: str, fn, cn, relation, angles_navigation, r1, r2, r_f, q_f, multy_antenna_send: bool, multy_antenna_take: bool, w_0: float, t: float, q1: None, q2: None):
    """Возвращает элемент матрицы Н в L_z^(c/d) строк и 6/13 столбцов.
    Столбец элемента матрицы H определяет только i_x. Остальные параметры определяют строку.
    :param i_x: Номер чипсата, у которого оценивается вектор-состояние (N вширь всей матрицы H)
    :param i_y: Номер чипсата/кубсата, от которого идёт сигнал (M / N(N-1) ввысь половины матрицы H)
    :param i_n: Номер чипсата, к которому идёт сигнал N ввысь элемента матрицы H^cf/H^ff)
    :param i: Номер 1-го КА
    :param j: Номер 2-го КА
    :param fn: Количество чипсатов
    :param cn: Количество кубсатов
    :param i_all: Количество антенн у 1-го КА
    :param j_all: Количество антенн у 2-го КА
    :param relation: Блок кубат-чипсат (cf) или чипсат-чипсат (ff) 
    :param angles_navigation: Оценивается ли вращательное движение
    :param r1: Положение 1-го КА
    :param r2: Положение 2-го КА
    :param r_f: Положения чипсатов
    :param q_f: Вектор-часть кватернионов ориентации чипсатов
    :param multy_antenna_send: Раскладывается ли сигнал при отправке
    :param multy_antenna_take: Раскладывается ли сигнал при принятии
    :param w_0: Угловая скорость вращения ОСК
    :param t: Текущее время
    :param q1: Кватернион 1-го КА (опционально)
    :param q2: Кватернион 2-го КА (опционально)
    """

    ff_sequence = []  # Последовательность номеров непустых столбцов, длина ff_sequence - кол-во строк нижней подматицы
    for i_f1 in range(fn):
        for i_f2 in range(i_f1):
            if i_f1 != i_f2:
                ff_sequence += [[i_f1, i_f2]]

    r1_x, r1_y, r1_z = r1
    r2_x, r2_y, r2_z = r2
    r12x = r1_x - r2_x
    r12y = r1_y - r2_y
    r12z = r1_z - r2_z
    r12 = sqrt(r12x**2 + r12y**2 + r12z**2)
    if q1 is not None:
        q1_x, q1_y, q1_z = q1
        q1_0 = 1 - sqrt(q1_x**2 + q1_y**2 + q1_z**2)
    if q2 is not None:
        q2_x, q2_y, q2_z = q2
        q2_0 = 1 - sqrt(q2_x**2 + q2_y**2 + q2_z**2)
    ω_0 = w_0

    to_replace
    
    if angles_navigation:
        swt = sin(t*w_0)
        cwt = cos(t*w_0)
        s1_x_r12 = s1_11*(r12x) + s1_12*(r12y) + s1_13*(r12z)
        s1_y_r12 = s1_21*(r12x) + s1_22*(r12y) + s1_23*(r12z)
        s1_z_r12 = s1_31*(r12x) + s1_32*(r12y) + s1_33*(r12z)
        s2_x_r12 = s2_11*(r12x) + s2_12*(r12y) + s2_13*(r12z)
        s2_y_r12 = s2_21*(r12x) + s2_22*(r12y) + s2_23*(r12z)
        s2_z_r12 = s2_31*(r12x) + s2_32*(r12y) + s2_33*(r12z)
        s1_r12 = sqrt((s1_x_r12)**2 + (s1_y_r12)**2 + (s1_z_r12)**2)
        s2_r12 = sqrt((s2_x_r12)**2 + (s2_y_r12)**2 + (s2_z_r12)**2)
        s1_r12_2 = ((s1_x_r12)**2 + (s1_y_r12)**2 + (s1_z_r12)**2)
        s2_r12_2 = ((s2_x_r12)**2 + (s2_y_r12)**2 + (s2_z_r12)**2)
    
        s1_xyx = sqrt((s1_x_r12)**2 + (s1_y_r12)**2 + (s1_x_r12)**2)
        s1_xy = sqrt((s1_x_r12)**2 + (s1_y_r12)**2)
        s1_yz = sqrt((s1_y_r12)**2 + (s1_z_r12)**2)
        s1_xz = sqrt((s1_x_r12)**2 + (s1_z_r12)**2)
        s2_xyx = sqrt((s2_x_r12)**2 + (s2_y_r12)**2 + (s2_x_r12)**2)
        s2_xy = sqrt((s2_x_r12)**2 + (s2_y_r12)**2)
        s2_yz = sqrt((s2_y_r12)**2 + (s2_z_r12)**2)
        s2_xz = sqrt((s2_x_r12)**2 + (s2_z_r12)**2)
    
        s1_xyx_2 = ((s1_x_r12)**2 + (s1_y_r12)**2 + (s1_x_r12)**2)
        s1_xy_2 = ((s1_x_r12)**2 + (s1_y_r12)**2)
        s1_yz_2 = ((s1_y_r12)**2 + (s1_z_r12)**2)
        s1_xz_2 = ((s1_x_r12)**2 + (s1_z_r12)**2)
        s2_xyx_2 = ((s2_x_r12)**2 + (s2_y_r12)**2 + (s2_x_r12)**2)
        s2_xy_2 = ((s2_x_r12)**2 + (s2_y_r12)**2)
        s2_yz_2 = ((s2_y_r12)**2 + (s2_z_r12)**2)
        s2_xz_2 = ((s2_x_r12)**2 + (s2_z_r12)**2)
    
        c_s1_x = cos(pi*(s1_x_r12)/(2*s1_r12))
        c_s1_y = cos(pi*(s1_y_r12)/(2*s1_r12))
        c_s1_z = cos(pi*(s1_z_r12)/(2*s1_r12))
        c_s2_x = cos(pi*(s2_x_r12)/(2*s2_r12))
        c_s2_y = cos(pi*(s2_y_r12)/(2*s2_r12))
        c_s2_z = cos(pi*(s2_z_r12)/(2*s2_r12))

        q1_y_z2 = -2*q1_y**2 - 2*q1_z**2 + 1
        q1_x_z2 = -2*q1_x**2 - 2*q1_z**2 + 1
        q1_0z_xy1 = 2*q1_0*q1_z + 2*q1_x*q1_y
        q1_0z_xy2 = -2*q1_0*q1_z + 2*q1_x*q1_y
        q1_0y_xz1 = 2*q1_0*q1_y + 2*q1_x*q1_z
        q1_0x_yz2 = -2*q1_0*q1_x + 2*q1_y*q1_z

        q2_y_z2 = -2*q2_y**2 - 2*q2_z**2 + 1
        q2_x_z2 = -2*q2_x**2 - 2*q2_z**2 + 1
        q2_0z_xy1 = 2*q2_0*q2_z + 2*q2_x*q2_y
        q2_0z_xy2 = -2*q2_0*q2_z + 2*q2_x*q2_y
        q2_0y_xz1 = 2*q2_0*q2_y + 2*q2_x*q2_z
        q2_0x_yz2 = -2*q2_0*q2_x + 2*q2_y*q2_z
    
        c_s1_xr12_xyx = cos(pi*(s1_x_r12)/(2*s1_xyx))
        t001 = -2*cwt*q1_0z_xy1 + 2*q1_y_z2*swt  
        t002 = 2*cwt*q1_x_z2 + 2*q1_0z_xy1*swt   
        t003 = 2*cwt*s1_22 + 2*q1_0y_xz1*swt   
        t004 = -2*cwt*q2_0z_xy1 + 2*q2_y_z2*swt   
        t005 = 2*cwt*q2_x_z2 + 2*q2_0z_xy1*swt   
        t006 = 2*cwt*s2_22 + 2*q2_0y_xz1*swt 
