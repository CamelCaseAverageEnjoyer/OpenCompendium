# ---> в _objects.py

class ControlParams:
    """Глобальные переменные и параметры для управления
    kw, ka  - параметры ПД-регулятора
    Bprev   - поле в первый момент времени для алгоритма Bdot
    """
    def __init__(self, w02: float, if_Bprev: bool = False, F:np.ndarray = None, A:np.ndarray = None):
        self.kw = 40 / w02
        self.ka = 10

        if if_Bprev:
            self.kw = 5e6
            Bmag=Inclined_field(0,[1 0 0]');
            Bmag=F.T @ Bmag;
            self.Bprev = A @ Bmag;
