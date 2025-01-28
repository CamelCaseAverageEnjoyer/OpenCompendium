

def h_matrix(t, v, f, c, r_f, r_c, q_f, q_c: list, return_template: bool = False):
    '''Возвращает матрицу частных производных Н.
    :param c_ant: Количество антенн у кубсата
    :param f_ant: Количество антенн у чипсата
    :param fn: Количество чипсатов
    :param cn: Количество кубсатов
    :param angles_navigation: Оценивается ли вращательное движение
    :param r_f: Положения чипсатов
    :param r_c: Положения кубсатов
    :param multy_antenna_send: Раскладывается ли сигнал при отправке
    :param multy_antenna_take: Раскладывается ли сигнал при принятии
    :param w_0: Угловая скорость вращения ОСК
    :param t: Текущее время
    :param q_f: Вектор-часть кватернионов чипсатов (опционально)
    :param q_c: Вектор-часть кватернионов кубсатов (опционально)
    :return: Матрица частных производных H. Отображение состояния в измерения
    ''' 
    from sympy import var

    fn = f.n
    cn = c.n
    c_g = v.GAIN_MODEL_C
    f_g = v.GAIN_MODEL_F
    c_ant = v.N_ANTENNA_C
    f_ant = v.N_ANTENNA_F
    angles_navigation = v.NAVIGATION_ANGLES
    multy_antenna_send = v.MULTI_ANTENNA_SEND
    multy_antenna_take = v.MULTI_ANTENNA_TAKE
    w_0 = v.W_ORB

    ff_sequence = []  # Последовательность номеров непустых столбцов, длина ff_sequence - кол-во строк нижней подматицы
    for i_f1 in range(fn):
        for i_f2 in range(i_f1):
            if i_f1 != i_f2:
                ff_sequence += [[i_f1, i_f2]]

    # >>>>>>>>>>>> Верхняя подматрица <<<<<<<<<<<<
    H_cd = None
    for i_c in range(cn):
        row = []
        for i_f in range(fn):
            if return_template:
                row.append(var(f'cd_' + str(i_c) + '^' + str(i_f)))
            else:
                row.append(h_element(i_x=None, i_y=None, i_n=None, i=i_c, j=i_f, gm_1=c_g, gm_2=f_g, fn=fn, cn=cn, relation='cd', angles_navigation=angles_navigation, r_f=r_f, r1=r_c[i_c], r2=r_f[i_f], q_f=q_f, q1=q_c[i_c], q2=q_f[i_f], multy_antenna_send=multy_antenna_send, multy_antenna_take=multy_antenna_take, w_0=w_0, t=t))
                
        row = block_diag(*row)
        H_cd = row if H_cd is None else vstack([H_cd, row])

    # >>>>>>>>>>>> Нижняя подматрица <<<<<<<<<<<<
    H_dd = None
    for i_y in range(len(ff_sequence)):  # то же самое, что range(int(fn*(fn-1)/2)):
        row = []
        for i_x in range(fn):
            if i_x in ff_sequence[i_y]:
                i_1, i_2 = ff_sequence[i_y] if i_x == ff_sequence[i_y][0] else ff_sequence[i_y][::-1]  # Я тут не перепутал????????
                if return_template:
                    row.append(var(f'dd_' + str(i_1) + '^' + str(i_2)))
                else:
                    row.append(h_element(i_x=i_x, i_y=i_y, i_n=i_f, i=i_1, j=i_2, gm_1=f_g, gm_2=f_g, fn=fn, cn=cn, relation=relation, angles_navigation=angles_navigation, r_f=r_f, r1=r_f[i_1], r2=r_f[i_2], q_f=q_f, q1=q_f[i_1], q2=q_f[i_2], multy_antenna_send=multy_antenna_send, multy_antenna_take=multy_antenna_take, w_0=w_0, t=t))
                    # ОСТАНОВИЛСЯ ТУТ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    

        row = bmat(row)
        H_dd = row if H_dd is None else vstack([H_dd, row])   


    
    for i_y in range(len(ff_sequence)):  # то же самое, что range(int(fn*(fn-1)/2)):
        row = []
        for i_x in range(fn):
            row_col = []
            for i_f in range(1):
                i_1, i_2 = ff_sequence[i_y]

                if return_template:
                    tmp_row_col += [var(relation + '_' + str(i_1) + '^' + str(i_2))]
                else:
                    h = h_element(i_x=i_x, i_y=i_y, i_n=i_f, i=i_1, j=i_2, gm_1=f_g, gm_2=f_g, fn=fn, cn=cn, relation=relation, angles_navigation=angles_navigation, r_f=r_f, r1=r_f[i_1], r2=r_f[i_2], q_f=q_f, q1=q_f[i_1], q2=q_f[i_2], multy_antenna_send=multy_antenna_send, multy_antenna_take=multy_antenna_take, w_0=w_0, t=t)

            row += [vstack(row_col)]
        row = bmat(row)
        H_dd = row if H_dd is None else vstack([H_dd, row])

    # >>>>>>>>>>>> Компановка <<<<<<<<<<<<
    return vstack([H_cd, H_dd]) if H_dd is not None else H_cd
