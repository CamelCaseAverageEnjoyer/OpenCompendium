# ---> в 00_tiny_functions.py

def trajectory(w: float, C: list, t:float) -> np.ndarray: 
    return np.array([-3*C[0]*w*t + 2*C[1]*np.cos(w*t) - 2*C[2]*np.sin(w*t) + C[3],
                     C[4]*np.sin(w*t) + C[5]*np.cos(w*t),
                     2*C[0] + C[1]*np.sin(w*t) + C[2]*np.cos(w*t),
                     -3*C[0]*w - 2*C[1]*w*np.sin(w*t) - 2*C[2]*w*np.cos(w*t),
                     C[4]*w*np.cos(w*t) - C[5]*w*np.sin(w*t),
                     C[1]*w*np.cos(w*t) - C[2]*w*np.sin(w*t)])
