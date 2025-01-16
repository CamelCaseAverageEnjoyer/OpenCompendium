from __init__ import *
from sympy import *
import matplotlib.pyplot as plt
import numpy as np

# >>>>>>>>>>>>>>>>>>>> OBSERVABILITY <<<<<<<<<<<<<<<<<<<< (функции)
def save_reports(report_list: list, filename: str) -> None:
    """Сохраняет в компактном виде несколько выводов в один
    :param report_list: Строки вывода, запрашиваемые к объединению
    :param filename: Название текстовика, в котором будет храниться информация"""
    common_report = report_list[0]
    for i in range(len(report_list) - 1):
        common_report += ("\n" + "-" * 100)*2 + "\n" + report_list[i + 1]
    f = open("storage/observability_reoprt_" + filename + ".txt", "w")
    f.write(common_report)
    f.close()

def read_reports(filename: str) -> str:
    """Считывает выводы, сохранённые в один файл
    :param filename: Название текстовика, в котором хранится информация"""
    f = open("storage/observability_reoprt_" + filename + ".txt", "r")
    common_report = f.read()
    f.close()
    return common_report

def print_spectrm_rank(J_numb):
    for tol in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-20]:
        print(f"Ранг матрицы: {np.linalg.matrix_rank(J_numb, tol=tol)} (tol={tol})")

# >>>>>>>>>>>>>>>>>>>> OBSERVABILITY <<<<<<<<<<<<<<<<<<<< (переменные)
def get_state_vector(func, obj: str, n: int = 1):
    kw = {'n': 3, 'numb': False}
    if func == kf.get_func:
        kw['t'] = t
    return ([func(name=f'r_{i}^{obj}', **kw) for i in range(n)],
            [func(name=f'v_{i}^{obj}', **kw) for i in range(n)],
            [kf.vec2quat(func(name=f'q_{i}^{obj}', **kw)) for i in range(n)],
            [func(name=f'ω_{i}^{obj}', **kw) for i in range(n)])

def init_symbol_params():
    o = kf.init()

    # Запись численных значений
    num_params = {'h': o.v.HEIGHT, 
                  'r0': o.v.ORBIT_RADIUS,
                  'ω0': o.v.W_ORB,
                  'v0': o.v.V_ORB,
                  'μ': o.v.MU,
                  'Jd': o.f.J,
                  'md': o.f.mass, 
                  'Cd': o.f.c_resist,
                  'sd': o.f.size,
                  'Jc': o.c.J,
                  'mc': o.c.mass, 
                  'Cc': o.c.c_resist,
                  'sc': o.c.size,
                  'ρ': kf.get_atm_params(v=o.v, h=o.v.HEIGHT)[0],
                 }
    print(f"Высота орбиты: {int(o.v.HEIGHT // 1e3)} км\nПериод орбиты: {round((2*np.pi/o.v.W_ORB) / 3600, 2)} часов\nПлотность атмосферы: {num_params['ρ']} кг/м³")

    # Подстановка символьных переменных
    t, ω, μ, ρ, r_orb, v_orb = var("t ω_0 μ ρ r_0 v_0")
    o.v.ORBIT_RADIUS = r_orb
    o.v.V_ORB = v_orb
    o.v.W_ORB = ω
    o.v.MU = μ
    
    # Те, которые не передаются
    o.v.dT = var('dt')
    
    J = [kf.get_vars("J^d", 3, numb=False), kf.get_vars("J^c", 3, numb=False)]
    o.f.J = diag(J[0][0], J[0][1], J[0][2])
    o.c.J = diag(J[1][0], J[1][1], J[1][2])
    o.f.mass, o.c.mass = var('m_d m_c')
    o.f.c_resist, o.c.c_resist = var('C_d C_c')
    o.f.size = kf.get_vars(name='s^d', n=2)
    o.c.size = kf.get_vars(name='s^c', n=2)

    return o, num_params, t, ω, μ, ρ, r_orb, v_orb
    
