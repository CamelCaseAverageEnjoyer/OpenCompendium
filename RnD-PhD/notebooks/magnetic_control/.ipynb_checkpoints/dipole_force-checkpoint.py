# ---> в _tiny_functions.py

def dipole_force(mu_0, r: np.ndarray, mu, Mu):
    """Это магнитный или электрический? Че делать с Mu'*mu"""
    R = np.linalg.norm(r)
    F = (3 * mu_0 / (4 * np.pi * R**5)) * (Mu'*mu*r + Mu'*r*mu + mu'*r*Mu - (5/R**2)*(Mu'*r)*(mu'*r)*r) ;
    return F
