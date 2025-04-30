"""
Единая система уравнений из kiam-formation. Смотри PyCharm!
"""
import sys  

# path = "/home/kodiak/Desktop/OpenCompendium/RnD-PhD/srs"
path = "../srs"
sys.path.insert(0, path)

import kiamformation as kf

def check_path():
    from os import listdir
    print(sorted(listdir(path)))
