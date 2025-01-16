# ---> в _tiny_functions.py

def mag_field(mu_0, r, Mu):
    """
    :return: B
    """ 
    R = np.linalg.norm(r)
    return (mu_0 / (4 * np.pi * R**5)) * (3 * Mu' * r * r - R**2 * Mu)
