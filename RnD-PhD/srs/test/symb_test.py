"""Проверка символьно-численного полиморфизма"""
import unittest
from srs.kiamformation.flexmath import *
from srs.kiamformation.cosmetic import my_print

class MyTests(unittest.TestCase):
    """Методы:
    self.assertEqual
    self.assertTrue
    self.assertFalse

    s = 'hello world'
    self.assertEqual(s.split(), ['hello', 'world'])
    """
    def test_elementary_funcs(self):
        my_print("Проверка элементарных функций", color='c')
        a1, a2, a3, a4 = sympy.symbols("a_1 a_2 a_3 a_4")

        a = np.array([0, 1, 2, 3, 4])
        b = append([0, 1], [2, 3, 4])
        print(f"(append) 1: {a} = {b}")
        self.assertTrue((a == b).all())

        a = sympy.Matrix([a1, a2, a3, a4])
        b = append(sympy.Matrix([a1, a2]), sympy.Matrix([a3, a4]))
        print(f"(append) 2: {a} = {b}")
        self.assertTrue(a == b)

        a = (1 + 2 + 3 + 4) / 4
        b = mean([1, 2, 3, 4])
        print(f"(mean) 1: {a} = {b}")
        self.assertTrue(a == b)

        a = (1 + 2 + 3 + 4) / 4
        b = mean(np.array([1, 2, 3, 4]))
        print(f"(mean) 2: {a} = {b}")
        self.assertTrue(a == b)

        a = (a1 + a2 + a3 + a4) / 4
        b = mean(sympy.Matrix([a1, a2, a3, a4]))
        print(f"(mean) 3: {a} = {b}")
        self.assertTrue(a == b)


    def test_calc_funcs(self):
        from srs.kiamformation.dynamics import get_c_hkw, r_hkw, v_hkw
        my_print("Проверка рассчётных функций", color='c')
        r = np.random.uniform(-1, 1, 3)
        v = np.random.uniform(-1, 1, 3)
        w = np.random.uniform(-1, 1)
        C = get_c_hkw(r=r, v=v, w=w)
        r1 = r_hkw(C=C, w=w, t=0)
        print(f"(hkw) 1: {r} = {r1}")
        self.assertTrue((np.abs(r - r1) < 1e-10).all())
        v1 = v_hkw(C=C, w=w, t=0)
        print(f"(hkw) 2: {v} = {v1}")
        self.assertTrue((np.abs(v - v1) < 1e-10).all())

if __name__ == "__main__":
    unittest.main()
