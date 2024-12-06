### Орбитальная сборка фермы с аппаратами-сборщиками

![](res_hdr_3_3.gif)

---
#### <image src="https://cdn.icon-icons.com/icons2/548/PNG/512/1455554839_line-60_icon-icons.com_53339.png" width=15> Программные файлы проекта 
  
- main_non_interface.py [Запуск сборки]
- main_interface.py [Запуск tk-приложения]
- all_object.py [Инициализация класса со всеми переменными и функциями][Гуляет почти везде]

- tkinter_files
  - assembly0.py [Окно с вводом параметров]
  - assembly1.py [Окно с динамикой сборки]
  - information.py [Что я такое и как меня есть]
  - plotting0.py [Вызов функций для отображения красивых картинок][Пойдёт в статью]
  - testing0.py [Вызов функций тестировки][Не пойдёт в статью]
  - tk_functions.py [Вспомогательные функции]

- mylibs 
    - calculation_function.py [Основные алгоритмы физики движения и взаимодействия]
    - construction_function.py [Инициализация pandas-таблиц о конструкциях и аппаратах]
    - control_function.py [Алгоритмы управления]
    - im_sample.py [Функции голосового помощника]
    - plot_function.py [Функции отображения]
    - result_show_functions.py [Показательные эксперименты][Вызывается plotting0.py]
    - test_functions.py [Тестировка][Вызывается testing0.py]
    - tiny_functions.py [Вспомогательные функции]
  
---
Методы scipy.optimize.minimize:
- trust-constr:
  - Bounds, (min, max)
  - нелинейное программирование
  - направление и длина шага выбираются одновременно 
- SLSQP:
  - Bounds, (min, max), LinearConstraint, NonlinearConstraint, словари
  - без ограничений -> метод Ньютона 
- TNC:
  - Bounds, (min, max)
  - ограниченное число итераций 
- COBYLA:
  - LinearConstraint, NonlinearConstraint, словари
  - ограниченная оптимизация без градиентов

---
#### <image src="https://cdn.icon-icons.com/icons2/2596/PNG/512/fire_icon_155393.png" width=15> Гори оно синим пламенем
  
  <img src="https://media.giphy.com/media/209KCwPntVIHsr5iIy/giphy.gif" alt="gif"/>

