"""
Единая система уравнений из kiam-formation. Смотри PyCharm!
"""
import sys  

path = "/home/kodiak/PycharmProjects/NIR_FEMTO/srs"
sys.path.insert(0, path)

import kiamformation as kf

def check_path():
    from os import listdir
    print(sorted(listdir(path)))
