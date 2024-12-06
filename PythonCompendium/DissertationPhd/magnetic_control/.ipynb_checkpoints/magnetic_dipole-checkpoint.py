# ---> в _tiny_functions.py

def function magnetic_dipole(F, d, mu1) -> np.ndarray:
    """
    :param F:
    :param d:
    :param mu1:
    :return: MU
    """
    import sympy as sym
    mu_0 = 4*sym.pi*1e-7  # Проницаемость свободного пространства
    mu2_x, mu2_y, mu2_z = sym.var("mu2_x mu2_y mu2_z")
    eqn1 = sym.Eq((3*mu_0/(4*pi*(norm(d))^5))*((mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(1)+(mu1'*d)*mu2_x+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(1)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(1)), F)
    eqn2 = sym.Eq((mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(2)+(mu1'*d)*mu2_y+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(2)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(2), 0)
    eqn3 = sym.Eq((mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(3)+(mu1'*d)*mu2_z+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(3)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(3), 0)
    sol = nsolve([eqn1, eqn2, eqn3], [mu2_x mu2_y mu2_z])
    return np.array([sol.mu2_x, sol.mu2_y, sol.mu2_z])
