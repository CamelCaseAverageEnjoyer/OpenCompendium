# ---> в _tiny_functions.py

def mag_field(mu_0, r, Mu):
    """
    :param mu_0:
    :param r: Радиус-вектор в какой-то СК
    :param Mu:
    :return: B - напряжённость магнитного поля
    """
    R = np.linalg.norm(r)
    B = (mu_0/(4*np.pi*R**5))*(3*Mu*r**2 - R**2*Mu)
    return B