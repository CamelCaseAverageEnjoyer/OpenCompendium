# ---> в _tiny_functions.py

def Hill_equation(y: np.ndarray, W: float):
    """
    :return: 6-вектор dy    
    """
    return np.array([y[3], y[4], y[5], -2*W*y[5], -W**2*y[1], 2*W*y[3] + 3*W**2*y[2]])
