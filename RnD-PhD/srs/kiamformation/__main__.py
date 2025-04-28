"""Численное моделирование космической миссии с использованием чипсатов"""
if __name__ == '__main__':
    import sys
    import numpy as np
    import pandas as pd
    from interface import interface_window
    from config import init
    from warnings import simplefilter

    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    # Инициализация объектов
    o = init()
    np.set_printoptions(linewidth=300)

    # Интерфейс
    app, window = interface_window(o=o)
    sys.exit(app.exec_())

