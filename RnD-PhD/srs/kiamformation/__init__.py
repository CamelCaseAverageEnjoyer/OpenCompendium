"""kiam-formation project"""
import sys
sys.path.insert(1, f"{sys.path[0]}/kiamformation")
sys.path.insert(1, f"{sys.path[0]}/test")

from config import *
from cosmetic import *
from dynamics import *
from simulation import *
from gnc_systems import *
# from interface import *
from my_math import *
from my_plot import *
from primary_info import *
from spacecrafts import *
from flexmath import *
from H_matrix import *

my_print(f"Инициализация проекта kiam-formation", color="g")
