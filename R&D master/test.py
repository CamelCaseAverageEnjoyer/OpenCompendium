import unittest
from all_objects import *


def get_params_repulsion_config_1app(o_: AllProblemObjects, id_app: int = 0):
    return o_.a.target[id_app][0], o_.a.target[id_app][1], o_.a.target[id_app][2], \
        o_.a.flag_beam[id_app], o_.a.flag_start[id_app], o_.a.flag_fly[id_app], \
        o_.a.flag_hkw[id_app], o_.flag_vision[id_app], o_.t_reaction_counter

class TestStringMethods(unittest.TestCase):
    """def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)"""
    def test_repulse_app_config_1(self):
        """Сразу с места"""
        o = AllProblemObjects(if_any_print=False, if_testing_mode=False)
        params = get_params_repulsion_config_1app(o)

        o.repulse_app_config(0)
        o.remove_repulse_app_config(0)
        self.assertEqual(get_params_repulsion_config_1app(o), params)

    def test_repulse_app_config_2(self):
        """Сначала полетел, потом проверка в воздухе"""
        o = AllProblemObjects(if_any_print=False, if_testing_mode=False)
        _ = repulsion(o, 0, u_a_priori=np.array([0.000419018231378659, -0.01500023895830125, -0.0004078787979977080]))
        for _ in range(10):
            o.control_step(0)
        params = get_params_repulsion_config_1app(o)

        o.repulse_app_config(0)
        o.remove_repulse_app_config(0)
        self.assertEqual(get_params_repulsion_config_1app(o), params)

    def test_repulse_app_config_3(self):
        """Сначала полетел, потом типа прикрепился, затем проверка"""
        o = AllProblemObjects(if_any_print=False, if_testing_mode=False)
        _ = repulsion(o, 0, u_a_priori=np.array([0.000419018231378659, -0.01500023895830125, -0.0004078787979977080]))
        for _ in range(10):
            o.control_step(0)
        capturing(o=o, id_app=0)
        params = get_params_repulsion_config_1app(o)

        o.repulse_app_config(0)
        o.remove_repulse_app_config(0)
        self.assertEqual(get_params_repulsion_config_1app(o), params)

if __name__ == '__main__':
    unittest.main()
