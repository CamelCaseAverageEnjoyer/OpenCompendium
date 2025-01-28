

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
    
    cf_matrix = None
    ff_matrix = None
    for relation in ['cf', 'ff']:
        tmp_rows = None
        for i_y in range(cn if relation=='cf' else int(fn*(fn-1)/2)): # Что тут не так?
            tmp_row = []   
            for i_x in range(fn):
                tmp_row_col = []
                for i_f in range(fn if relation=='cf' else 1): # ВОТ ТУТ ТО СВИНЬЯ И ЗАРЫТА ДАААААААААААААААААААААААААААААААА 
                    if relation == 'cf':
                        i_1 = i_y   
                        i_2 = i_f   
                    else:
                        i_1 = ff_sequence[i_y][0] 
                        i_2 = ff_sequence[i_y][1] 
                    
                    if return_template:
                        tmp_row_col += [var(relation + '_' + str(i_1) + '^' + str(i_2))]
                    else:
                        h = h_element(i_x=i_x, i_y=i_y, i_n=i_f, i=i_1, j=i_2, 
                                      gm_1=c_g if relation=='cf' else f_g, 
                                      gm_2=f_g, 
                                      fn=fn, cn=cn, relation=relation, angles_navigation=angles_navigation, r_f=r_f, r1=r_c[i_1] if relation=='cf' else r_f[i_1], r2=r_f[i_2], q_f=q_f, q1=q_c[i_1] if relation=='cf' else q_f[i_1], q2=q_f[i_2], multy_antenna_send=multy_antenna_send, multy_antenna_take=multy_antenna_take, w_0=w_0, t=t)
                        tmp_row_col.append(h)

                tmp_row += [vstack(tmp_row_col)]

            tmp_row = bmat(tmp_row)
            tmp_rows = tmp_row if tmp_rows is None else vstack([tmp_rows, tmp_row])

        if relation == 'cf':
            cf_matrix = tmp_rows
        else:
            ff_matrix = tmp_rows

    if ff_matrix is not None:
        return vstack([cf_matrix, ff_matrix])
    else: 
        return cf_matrix