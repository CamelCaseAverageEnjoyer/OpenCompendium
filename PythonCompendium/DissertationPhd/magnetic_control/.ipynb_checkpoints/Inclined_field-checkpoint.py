# ---> в _tiny_functions.py
from datetime import timedelta

def Inclined_field(t: float, r: np.ndarray, B0, t0scal: datetime, k0, DEG_TO_RAD: float):
    """
    :param t:
    :param r:
    :param B0: был global КАКАЯ РАЗМЕРНОСТЬ
    :param t0scal: был global
    :param k0: был global
    :param DEG_TO_RAD: был global
    :retrun: Bmag
    """
    # Получаем текущее время
    tcur = t0scal + timedelta(milliseconds=round(1000 * t))
    # Вычисляем юлианскую дату в днях
    JD = juliandate(tcur.date())  # Сверено с комплексом
    
    # Перевод вектора направления диполя от гринвической системы к инерциальной текущей даты
	theta = (280.46061837 + 360.98564736629 * (JD - 2451545.0)) * DEG_TO_RAD
    
	k = [k0[0] * np.cos(theta) - k0[1] * np.sin(theta),
	     k0[0] * np.sin(theta) + k0[1] * np.cos(theta),
	     k0[2]]
    return B0 * (k'-3 * (k * r) * r)  # АААААААААААААААААААААААААААА?
