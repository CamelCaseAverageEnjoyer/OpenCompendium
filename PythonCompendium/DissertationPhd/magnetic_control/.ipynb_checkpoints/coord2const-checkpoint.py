# ---> в _tiny_functions.py

def coord2const(X: np.ndarray, w: float) -> list:
    """Функция констант С для ХКУ движения
    :param X: Фазовый 6-вектор
    :param w: Угловая скорость орбиты
    """
    return [2 * X[2] + X[3] / w,
            X[5] / w,
            -3 * X[2] - 2 * X[3] / w,
            X[0] - 2 * X[5] / w,
            X[4] / w,
            X[1]]

