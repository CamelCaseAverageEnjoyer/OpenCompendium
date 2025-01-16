# ---> в _objects.py
from datetime import datetime

class Definitions:
    """
    J, Jreal             - тензор инерции
    w02, w0, incl, F, p  - параметры орбиты
    t0scal               - начальный момент времени
    B0, k0               - параметры модели магнитного поля
    DEG_TO_RAD           - перевод градусов в радианы
    h_wheel              - кинетический момент маховика
    """
    def __init__(self):
        self.DEG_TO_RAD = np.pi / 180
        self.h_wheel = np.zeros(3)  # Маховик

        # Тензор инерции
        self.J = np.diag([0.01, 0.01, 0.007])  # "идеальный" тензор инерции из ТЗ - используется в модели движения в ФК
        # self.J = np.diag([3.600, 5.836, 2.468])
        # Ручная настройка
        # self.Jreal = np.diag([5836, 2468, 3600])  # "реальный" тензор инерции с неточностью 10% - используется в моделировании
        self.Jreal = self.J.copy()
        if (self.Jreal[0][0] + self.Jreal[1][1] < self.Jreal[2][2]) or 
           (self.Jreal[1][1] + self.Jreal[2][2] < self.Jreal[0][0]) or 
           (self.Jreal[2][2] + self.Jreal[0][0] < self.Jreal[1][1]):
            print(f"Ёлы палы шо с тензором инерции, посмотри в definitions!")
         
        # Параметры орбиты
        H = 340.  # Высота орбиты. Значения: 750 (ШАМАН)
        r_earth = 6371.  # Средний радиус Земли (ШАМАН)
        r = r_earth + H
        self.incl = 60 * np.pi / 180  # Наклонение 
        Omega0 = 0.  # Долгота восходящего узла
        omega = 0.  # Аргумент перицентра
        u0 = 0.  # Начальный аргумент широты
        self.p = r  # Параметр орбиты
        self.w02 = (398600.4415 / r**3)**5  # Орбитальная скорость определяется по высоте (ШАМАН)
        self.w0 = np.array([0, self.w02, 0])
        # Матрица перехода от ИСК(орбита) -> ИСК(Земля)
        self.F = np.array([[np.cos(Omega0)*np.cos(omega)-np.sin(Omega0)*np.sin(omega)*np.cos(self.incl), 
                            -np.cos(Omega0)*np.sin(omega)-np.sin(Omega0)*np.cos(omega)*np.cos(self.incl), 
                            np.sin(Omega0)*np.sin(self.incl)],
                           [np.sin(Omega0)*np.cos(omega)+np.cos(Omega0)*np.sin(omega)*np.cos(self.incl), 
                            -np.sin(Omega0)*np.sin(omega)+np.cos(Omega0)*np.cos(omega)*np.cos(self.incl), 
                            -np.cos(Omega0)*np.sin(self.incl),
                           [np.sin(omega)*np.sin(self.incl), 
                            np.cos(omega)*np.sin(self.incl), 
                            np.cos(self.incl)]]])

        # Параметры наклонного диполя
        mu = 7.812e6
        self.B0 = mu / self.p**3
        lambda0 = -71.88 * self.DEG_TO_RAD  # Углы ориентации диполя относительно Земли
        delta0 = 10.26 * self.DEG_TO_RAD
        self.k0 = np.array([np.cos(lambda0) * np.sin(delta0),
                            np.sin(lambda0) * np.sin(delta0),
                            np.cos(delta0)])  # Единичный вектор диполя относительно Земли

        # Начальный момент времени
        self.t0scal = datetime(year=2015, month=1, day=1, hour=0, minute=0, second=0)
