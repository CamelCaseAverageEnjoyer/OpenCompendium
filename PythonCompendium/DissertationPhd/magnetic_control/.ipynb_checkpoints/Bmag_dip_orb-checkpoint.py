# ---> в _tiny_functions.py

def Bmag_dip_orb(u: float, incl: float, B0, w02: float) -> tuple:
    """ Расчёты ведутся в ОСК
    :param incl: был global
    :param B0: был global
    :param w02: был global
    """
    Bdip = np.array([np.sin(incl)*np.cos(u), np.cos(incl), -2*np.sin(incl)*np.sin(u)])
    dBdip = np.array([-w02*np.sin(incl)*np.sin(u), 0, -2*w02*np.sin(incl)*np.cos(u)])

    # B=B0*.5*(1+(1+3*sin(incl)^2)^.5)*Bdip.T
    B = B0 * Bdip'  # ЧТО ЗДЕСЬ ПРОИСХОДИТ, В0 ЭТО ЧИСЛО ИЛИ ВЕКТОР?
    dB = B0 * dBdip'  # А ВЫХОДНОЙ РЕЗАЛТ ЭТО ЧИСЛА ИЛИ ВЕКТОРА?
    return B, dB
