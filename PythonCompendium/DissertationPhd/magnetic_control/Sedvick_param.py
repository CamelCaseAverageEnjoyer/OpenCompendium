# ---> в 00_tiny_functions.py

def Sedvick_param(parameters):
    """
    :param parameters: (r0, i0, i1, i2, z0, dz0)
    :return: (n, c, q, l, phi)
    """
    J2 = 1.082e-3
    mu = 3.986e14
    R_Earth = 6.378245e6

    r0, i0, i1, i2, z0, dz0 = parameters

    c = np.sqrt(1 + 3 * J2 * R_Earth**2 * (1 + 3 * np.cos(2*i0)) / 8 / r0**2)
    n = np.sqrt(mu / r0**3)
    k = n * c + 3 * J2 * R_Earth**2 * np.cos(i0)**2 / 2 / r0**2
    i2 = dz0 / k / r0 + i1

    dW0 = z0 / r0 / np.sin(i0)

    q = n*c + 3*n*J2*R_Earth**2/2/r0**2*(np.cos(i2)**2-((cos(i1)-cos(i2))*(tan(i1)^(-1)*sin(i2)*cos(dW0)-cos(i2)))/(sin(dW0)^2+(tan(i1)^(-1)*sin(i2)-cos(i2)*cos(dW0))^2));
    l=-3*n*J2*R_Earth^2/2/r0*((cos(i1)-cos(i2))*sin(i1)*sin(i2)*sin(dW0))/sqrt(1-(cos(i1)*cos(i2)+sin(i1)*sin(i2)*cos(dW0))^2);

    a=roots([l^2; -2*dz0*l; q^2*z0^2+dz0^2; 0; - q^2*z0^2]);
    phi=asin(min(abs(a)));

    return n, c, q, l, phi

