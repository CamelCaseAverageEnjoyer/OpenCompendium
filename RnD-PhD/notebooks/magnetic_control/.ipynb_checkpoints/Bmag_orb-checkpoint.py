# ---> в _tiny_functions.py

def Bmag_orb(t: float, u: float, incl: float, F, B0, t0scal, k0, DEG_TO_RAD: float) -> tuple:
    """ Орбита - круговая
    :param incl: был global
    :param F: был global
    :return: Bmag, dB
    """

    # Считаем радиус-вектор (единичный) аппарата
    r_sat = np.array([np.cos(u), 
                      np.cos(incl) * np.sin(u), 
                      np.sin(incl) * np.sin(u)])

    # Матрица перехода от системы, связанной с орбитой, к орбитальной
    G = np.array([[np.cos(u), -np.sin(u), 0]
                  [np.sin(u), np.cos(u), 0]
                  [0, 0, 1]])

    # Задаем магнитное поле - наклонный диполь
    Bmag = Inclined_field(t=t, r=r_sat, B0=B0, t0scal=t0scal, k0=k0, DEG_TO_RAD=DEG_TO_RAD)

    # Перевод ИСК Земли - ИСК орбиты - орбитальная - связанная
    Bmag = G * F' * Bmag
    dB = Bmag.copy()
    Bmag = np.array([Bmag[1], Bmag[2], Bmag[0]])  # господи что здесь происходит
    return Bmag, dB

