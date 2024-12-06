import numpy as np
import copy
import matplotlib.pyplot as plt


def package_beams(N, h):
    ast = 2 * h / np.sqrt(3)
    x = np.zeros(N)
    y = np.zeros(N)
    flag = 0
    count = 0
    max_count = 0
    for i in range(N - 1):
        if flag > 6:
            flag = 0
            max_count += 1
        if flag == 0:  # вверх, новый уровень
            x[i + 1] = x[i]
            y[i + 1] = y[i] + ast
            count = 0
        if flag == 1:  # влево вниз
            x[i + 1] = x[i] - h
            y[i + 1] = y[i] - ast / 2
        if flag == 2:  # вниз
            x[i + 1] = x[i]
            y[i + 1] = y[i] - ast
        if flag == 3:  # вправо вниз
            x[i + 1] = x[i] + h
            y[i + 1] = y[i] - ast / 2
        if flag == 4:  # вправо вверх
            x[i + 1] = x[i] + h
            y[i + 1] = y[i] + ast / 2
        if flag == 5:  # вверх
            x[i + 1] = x[i]
            y[i + 1] = y[i] + ast
        if flag == 6:  # влево вверх
            if count > 0:
                x[i + 1] = x[i] - h
                y[i + 1] = y[i] + ast / 2
            else:
                flag = 0
                max_count += 1
                x[i + 1] = (x[i] - h)
                y[i + 1] = (y[i] + ast / 2) + ast

        if count == 0:
            flag += 1
            count = max_count
        else:
            count -= 1
    return x, y, max_count + 2


class Structure(object):
    def __init__(self, choice: str = '1', complete: bool = False, floor: int = 5, extrafloor: int = 0, mass_per_length: float = 1.,  # mass_check
                 testing=False):
        if floor < 1:
            raise "Поменяй параметры конструкции: floor должен быть равным 1 или больше"
        self.floor = floor
        self.extrafloor = extrafloor
        self.choice = choice
        self.container_length = 12.
        self.h = 0.15
        self.lvl = 0
        self.x_start = 0.6
        seq_truss_algo = [[1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 5],
                          [5, 6, 7, 8, 8, 5, 6, 7, 6, 7, 8, 8]]
        loc_truss_algo = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]

        def check_length():
            for i in range(self.n_beams):
                if self.length[i] < 1e-2:
                    print(f"Внимание! Стержни нулевой длины! i:{i}/{self.n_beams} id:{self.id[i]} len:{self.length[i]}")

        def check_plot():
            if testing:
                for k in range(len(self.r1)):
                    plt.plot([self.r1[k][0], self.r2[k][0]], [self.r1[k][2], self.r2[k][2]], c='navy')
                plt.axis('equal')
                plt.show()

        if choice == '0':
            self.n_beams = 1
            self.n_nodes = 2
            self.mass = np.array([0.])
            self.length = np.array([5.])
            self.id = np.array([0])
            self.id_node = np.array([np.array([0, 1])])
            self.r1 = np.array([np.array([1., 0., 0.])])
            self.r2 = np.array([np.array([6., 0., 0.])])
            self.flag = np.array([np.array([1, 1])])
            self.r_st = np.array([np.zeros(3)])

        if choice == '1':
            floor = 3
            self.n_beams = 24
            ast = 2 * self.h / np.sqrt(3)
            self.flag = np.array([np.array([1, 1])] * 3 + [np.array([0, 0])] * 21)
            x_st = [-self.x_start for _ in range(self.n_beams)]
            y_st = [2 * self.h, 3 * self.h, 3 * self.h, 2 * self.h, 1 * self.h, -1 * self.h, -2 * self.h, -2 * self.h,
                    -2 * self.h, -1 * self.h, 0, 1 * self.h, 2 * self.h,
                    2 * self.h, 2 * self.h, 1 * self.h, 0, -1 * self.h, -1 * self.h, 0, 1 * self.h, 1 * self.h, 0, 0]
            z_st = [-2 * ast, -0.5 * ast, 0.5 * ast, 2 * ast, 2.5 * ast, 1.5 * ast, 1 * ast, 0, -1 * ast,
                    -1.5 * ast, -2 * ast, -1.5 * ast, -1 * ast, 0, 1 * ast, 1.5 * ast, 2 * ast, 0.5 * ast,
                    -0.5 * ast, -1 * ast, -0.5 * ast, 0.5 * ast, 1 * ast, 0]
            self.r_st = np.array([np.array([x_st[i], y_st[i], z_st[i]]) for i in range(self.n_beams)])

        if choice in ['1', '2']:
            self.n_beams = 3 + floor * 3 + (floor - 1) * 6
            self.n_nodes = 1 + 3 * floor
            self.h = 0.25
            a_beam = 5.0
            r_inscribed = a_beam / 2 / np.sqrt(3)
            r_circumscribed = a_beam / np.sqrt(3)
            x_ground_floor = a_beam * np.sqrt(2 / 3)

            y_st, z_st, self.lvl = package_beams(self.n_beams, self.h)
            r = np.array([np.zeros(3) for _ in range(self.n_nodes)])  # Beam coordinates; r[0,:] = [x,y,z]
            r[1] = np.array([x_ground_floor, 0., r_circumscribed])
            r[2] = np.array([x_ground_floor, -a_beam / 2, -r_inscribed])
            r[3] = np.array([x_ground_floor, a_beam / 2, -r_inscribed])
            r1 = [r[0], r[0], r[0], r[1], r[1], r[2]]
            r2 = [r[1], r[2], r[3], r[2], r[3], r[3]]
            id_node_1 = [0, 0, 0, 1, 1, 2]
            id_node_2 = [1, 2, 3, 2, 3, 3]
            sequence = [[1, 2, 3, 1, 2, 3, 4, 5, 6], [4, 5, 6, 6, 4, 5, 6, 4, 5]]
            for i in range(floor - 1):
                r[3 * i + 4] = np.array([x_ground_floor + (i + 1) * a_beam, 0., r_circumscribed])
                r[3 * i + 5] = np.array([x_ground_floor + (i + 1) * a_beam, -a_beam / 2, -r_inscribed])
                r[3 * i + 6] = np.array([x_ground_floor + (i + 1) * a_beam, a_beam / 2, -r_inscribed])
                for j in range(len(sequence[0])):
                    id_1, id_2 = 3 * i + sequence[0][j], 3 * i + sequence[1][j]
                    r1.append(r[id_1])
                    r2.append(r[id_2])
                    id_node_1.append(id_1)
                    id_node_2.append(id_2)
                check_plot()

            self.id = np.arange(self.n_beams)
            self.id_node = np.array([np.array([id_node_1[i], id_node_2[i]]) for i in range(self.n_beams)])
            self.r1 = np.array(r1)
            self.r2 = np.array(r2)
            self.length = np.array([np.linalg.norm(np.array(self.r1[i]) - np.array(self.r2[i]))
                                    for i in range(self.n_beams)])
            self.container_length = np.max(self.length) + 0.5
            self.mass = self.length * mass_per_length
            self.mass = self.length * mass_per_length
            if choice != '1':
                self.flag = np.array([np.array([int(complete), int(complete)]) for _ in range(self.n_beams)])
                self.r_st = np.array([np.array([-self.x_start - self.container_length + self.length[i],
                                                y_st[i], z_st[i]]) for i in range(self.n_beams)])

        if choice == '3':
            length = 5
            self.n_beams = 84  # Beams
            self.n_nodes = 32  # Nodes
            self.x_start = 0.4
            R = length * 6
            a = 2 * np.arcsin(length / 2 / R)
            n_around = 3
            angle = np.linspace(a, a * n_around, n_around)
            r = [[R * (1 - np.cos(angle[i])) for i in range(n_around)],
                 [R * np.sin(angle[i]) for i in range(n_around)],
                 [0. for _ in range(n_around)]]
            for j in range(5):
                r[0] += r[0][0:3]
                r[1] += [r[1][i] * np.cos(2 * np.pi * (j + 1) / 6) for i in range(3)]
                r[2] += [r[1][i] * np.sin(2 * np.pi * (j + 1) / 6) for i in range(3)]
            for j in range(6):
                r[0] += r[0][1:3]
                r[1] += [r[1][i] * np.cos(2 * np.pi * (j + 1 / 2) / 6) for i in [1, 2]]
                r[2] += [r[1][i] * np.sin(2 * np.pi * (j + 1 / 2) / 6) for i in [1, 2]]
            r[0] += [0., 10.]
            r[1] += [0., 0.]
            r[2] += [0., 0.]
            id_node_1 = [30, 30, 30, 30, 30, 30, 15, 0, 3, 6, 9, 12, 15, 15, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 12, 12,
                         12, 15, 26, 16, 28, 1, 18, 4, 20, 7, 22, 10, 24, 13, 26, 16, 16, 16, 28, 1, 1, 1, 18, 4, 4, 4,
                         20, 7, 7, 7, 22, 10, 10, 10, 24, 13, 13, 13, 27, 17, 29, 2, 19, 5, 21, 8, 23, 11, 25, 14, 0, 3,
                         6, 9, 12, 15]
            id_node_2 = [0, 3, 6, 9, 12, 15, 0, 3, 6, 9, 12, 15, 16, 28, 28, 1, 18, 18, 4, 20, 20, 7, 22, 22, 10, 24,
                         24, 13, 26, 26, 16, 28, 1, 18, 4, 20, 7, 22, 10, 24, 13, 26, 27, 27, 17, 29, 29, 29, 2, 19, 19,
                         19, 5, 21, 21, 21, 8, 23, 23, 23, 11, 25, 25, 25, 14, 27, 17, 29, 2, 19, 5, 21, 8, 23, 11, 25,
                         14, 27, 31, 31, 31, 31, 31, 31]
            y_st, z_st, self.lvl = package_beams(self.n_beams, self.h)

            self.id = np.arange(self.n_beams)
            self.id_node = np.array([np.array([id_node_1[i], id_node_2[i]]) for i in range(self.n_beams)])
            self.r1 = np.array([np.array([r[0][id_node_1[i]], r[1][id_node_1[i]], r[2][id_node_1[i]]])
                                for i in range(self.n_beams)])
            self.r2 = np.array([np.array([r[0][id_node_2[i]], r[1][id_node_2[i]], r[2][id_node_2[i]]])
                                for i in range(self.n_beams)])

            self.flag = np.array([np.array([int(complete), int(complete)]) for _ in range(self.n_beams)])
            self.length = np.array([np.linalg.norm(self.r1[i] - self.r2[i]) for i in range(self.n_beams)])
            self.container_length = np.max(self.length) + 0.5
            self.mass = self.length * mass_per_length
            self.r_st = np.array([np.array([-self.x_start - self.container_length + self.length[i], y_st[i], z_st[i]])
                                  for i in range(self.n_beams)])

        if choice == '4':
            length = 5.
            beam_length_multiplier = 2
            self.beam_length_multiplier = beam_length_multiplier
            self.n_beams = 16
            self.h = 0.15
            self.id = list(range(self.n_beams))
            self.id_node_1 = [0, 0, 0, 0,
                              1, 1, 2, 3,
                              0, 0, 0, 0,
                              5 + floor * 4, 5 + floor * 4, 6 + floor * 4, 7 + floor * 4]
            self.id_node_2 = [1, 2, 3, 4,
                              4, 2, 3, 4,
                              5 + floor * 4, 6 + floor * 4, 7 + floor * 4, 8 + floor * 4,
                              8 + floor * 4, 6 + floor * 4, 7 + floor * 4, 8 + floor * 4]
            self.r1 = [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                       np.array([0., -length / 2, length]), np.array([0., length / 2, length]),
                       np.array([0., length / 2, length]), np.array([0., -length / 2, length]),
                       np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                       np.array([0., -length / 2, -length]), np.array([0., length / 2, -length]),
                       np.array([0., length / 2, -length]), np.array([0., -length / 2, -length])]
            self.r2 = [np.array([0., -length / 2, length]), np.array([0., length / 2, length]),
                       np.array([length, -length / 2, length]), np.array([length, length / 2, length]),
                       np.array([length, -length / 2, length]), np.array([0., -length / 2, length]),
                       np.array([length, length / 2, length]), np.array([length, length / 2, length]),
                       np.array([0., -length / 2, -length]), np.array([0., length / 2, -length]),
                       np.array([length, length / 2, -length]), np.array([length, -length / 2, -length]),
                       np.array([length, -length / 2, -length]), np.array([0., -length / 2, -length]),
                       np.array([length, length / 2, -length]), np.array([length, length / 2, -length])]

            def lcl_f(args):
                x, y, z, z_ind = args
                return np.array([x * length, length * (y - 1 / 2), z_ind * (i + 1 + z) * length])

            last_id = 0  # иначе оно конючит что может быть неинициализировано
            for i in range(floor):
                for z_index in [-1, 1]:
                    for j in range(12):
                        self.id += [int(self.n_beams + j + 12 * i + 1 + (12 * floor) * (0.5 - 0.5 * z_index))]
                    self.n_beams += 12
                    self.id_node_1 += [k + i * 4 + (floor + 1) * 4 * (0.5 - 0.5 * z_index) for k in seq_truss_algo[0]]
                    self.id_node_2 += [k + i * 4 + (floor + 1) * 4 * (0.5 - 0.5 * z_index) for k in seq_truss_algo[1]]
                    r_truss_algo = [lcl_f(loc_truss_algo[i] + [z_index]) for i in range(len(loc_truss_algo))]
                    self.r1 += [r_truss_algo[seq_truss_algo[0][k] - 1].copy() for k in range(12)]
                    self.r2 += [r_truss_algo[seq_truss_algo[1][k] - 1].copy() for k in range(12)]
                last_id = 8 + i * 4 + (floor + 1) * 4

            r_big_circle = length * (floor + 1)
            length *= beam_length_multiplier
            self.big_length = length
            self.r_big_circle = r_big_circle
            big_circle_floors = round(2 * np.pi * floor / beam_length_multiplier)
            phi = -np.pi * 2 / big_circle_floors
            rot_nods = np.array([np.array([0., -length / 2, r_big_circle + length]),
                                 np.array([0., length / 2, r_big_circle + length]),
                                 np.array([0., length / 2, r_big_circle]),
                                 np.array([0., -length / 2, r_big_circle])])
            rotation_matrix = np.array([[np.cos(phi), 0., np.sin(phi)],
                                        [0., 1., 0.],
                                        [-np.sin(phi), 0., np.cos(phi)]])

            for i in range(big_circle_floors):
                for j in range(12 * (1 + 2 * extrafloor)):
                    self.id += [self.n_beams]
                    self.n_beams += 1
                copies = copy.deepcopy(rot_nods)
                for k in range(4):
                    rot_nods[k] = rotation_matrix @ rot_nods[k]
                r_truss_algo = [copies[i] for i in range(4)] + [rot_nods[i] for i in range(4)]
                self.id_node_1 += [k + 1 + (last_id + 1 + i) * 4 for k in seq_truss_algo[0]]
                self.id_node_2 += [k + 1 + (last_id + 1 + i) * 4 for k in seq_truss_algo[1]]
                self.r1 += [r_truss_algo[seq_truss_algo[0][k] - 1].copy() for k in range(12)]
                self.r2 += [r_truss_algo[seq_truss_algo[1][k] - 1].copy() for k in range(12)]
                check_plot()

            y_st, z_st, self.lvl = package_beams(self.n_beams, self.h)
            self.n_nodes = len(self.id_node_1)
            self.id = np.array(self.id)
            self.r1 = np.array(self.r1)
            self.r2 = np.array(self.r2)
            self.id_node = np.array([np.array([int(self.id_node_1[i]), int(self.id_node_2[2])])
                                     for i in range(self.n_nodes)])
            self.flag = np.array([np.array([int(complete)] * 2) for i in range(self.n_beams)])
            self.length = np.array([np.linalg.norm(self.r1[i] - self.r2[i]) for i in range(self.n_beams)])
            self.container_length = np.max(self.length) + 0.1
            self.mass = self.length * mass_per_length
            self.r_st = np.array([[-self.x_start - self.container_length + self.length[i], y_st[i], z_st[i]]
                                  for i in range(self.n_beams)])

        if choice == '5':
            direction = ['+2', '-0', '+1', '+0', '-2', '-0', '-1', '-2', '+0', '-1', '+0', '+1', '+2', '-1', '+2',
                         '-0', '-2', '-0', '+2', '+2', '+1', '+0', '+1', '+0', '-1', '-2', '+1', '-2', '+1', '-0',
                         '-2', '-1', '-0', '+1', '+2', '-0', '-1', '+2', '+1', '+0', '+2', '+0', '-2', '+0', '+2',
                         '+0', '-1', '-2', '+1', '+0', '-1', '-2', '-0', '-1', '+2', '-1', '+2', '+1', '+0', '-2',
                         '-1', '-2', '-0', '-2', '+0', '+1', '-0', '+1', '+1', '+2', '+0', '-2', '-1', '+0', '-1',
                         '+2', '+1', '+2', '+1', '+2', '-0', '-1', '+0', '-1', '-2', '-1', '-2', '+0', '+1', '+2',
                         '-1', '+2', '-0']
            r1 = []
            r2 = []
            point = np.zeros(3)
            for i in range(len(direction)):
                r1 += [point]
                point[int(direction[i][1])] += 4. if direction[i][0] == '+' else -4.
                r2 += [point]
                check_plot()

            self.n_beams = 1
            self.n_nodes = 2
            self.id = np.array([0])
            self.mass = np.array([5.])
            self.id_node = np.array([[0, 1]])
            self.r1 = np.array([r2[len(r2) - 1]])
            self.r2 = np.array([r2[len(r2) - 1] + [0., 0., 1.]])
            self.length = np.array([1.])
            self.flag = np.array([np.zeros(2)])
            self.r_st = np.array([np.zeros(3)])

        # Post-init checks
        check_length()
        self.n = len(self.mass)

    def copy(self):
        s = Structure(choice=self.choice, floor=self.floor)
        s.n_beams = self.n_beams
        s.x_start = self.x_start
        s.n_nodes = self.n_nodes
        s.mass = copy.deepcopy(self.mass)
        s.length = copy.deepcopy(self.length)
        s.id_node = copy.deepcopy(self.id_node)
        s.r1 = copy.deepcopy(self.r1)
        s.r2 = copy.deepcopy(self.r2)
        s.flag = copy.deepcopy(self.flag)
        s.r_st = copy.deepcopy(self.r_st)
        s.container_length = self.container_length
        return s

    def call_possible_transport(self, taken_beams):
        """ Функция осуществляет последовательность сборки"""
        beams_to_take = np.array([])
        mask_non_fixed = [int(np.sum(self.flag[i]) == 0) for i in range(self.n_beams)]

        needed_number_nodes = np.zeros(self.n_nodes)
        current_number_nodes = np.zeros(self.n_nodes)
        mask_open_nodes = np.zeros(self.n_nodes)
        needed_number_nodes[self.id_node[0][0]] = 1   # Костыль на точку стыковки коснтрукции
        current_number_nodes[self.id_node[0][0]] = 1  # с грузовым контейнером

        for i in range(self.n_beams):
            needed_number_nodes[self.id_node[i][0]] += 1  # Сколько стержней приходят в узел
            needed_number_nodes[self.id_node[i][1]] += 1
            current_number_nodes[self.id_node[i][0]] += self.flag[i][0]  # Сколько стержней в узле находятся
            current_number_nodes[self.id_node[i][1]] += self.flag[i][1]

        for i in range(self.n_nodes):  # В каких узлах неполное кол-во стержней, но есть хоть один
            if (needed_number_nodes[i] - current_number_nodes[i] > 0) and (current_number_nodes[i] > 0):
                mask_open_nodes[i] = 1  # Основная маска

        for i in range(self.n_beams):
            if mask_non_fixed[i] > 0:  # Нетронутые со склада
                if mask_open_nodes[self.id_node[i][0]] + mask_open_nodes[self.id_node[i][1]] > 0:  # Надобность балки
                    beams_to_take = np.append(beams_to_take, self.id[i])

        i = 0
        while i < len(beams_to_take):  # Удалить те, которые уже взяты
            if beams_to_take[i] in taken_beams:
                beams_to_take = np.delete(beams_to_take, i)
                i -= 1
            i += 1

        return [int(i) for i in beams_to_take]


class Container(object):
    def __init__(self, s: Structure, choice='1'):
        self.choice = choice
        self.s = s
        self.r_around = 0

        if choice == '0':
            self.n = 1
            self.id = np.array([0, 1, 2])
            self.mass = np.array([5, 5, 5])
            self.diam = np.array([0.5, 0.5, 0.5])
            self.r1 = np.array([np.array([-6., 0., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 5.])])
            self.r2 = np.array([np.array([-1., 0., 0.]), np.array([6., 0., 0.]), np.array([0., 0., 10.])])
            self.flag_grab = np.array([True, False, False])

        if choice == '1':
            self.n = 7
            self.id = np.arange(self.n)
            self.mass = np.array([10.] + [1.] * 6)
            self.diam = np.array([5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            self.r1 = np.array([np.array([0.0, 0.0, 0.0]),
                                np.array([-s.x_start / 2, 1.0, 6.0]),
                                np.array([-s.x_start / 2, 1.0, 4.0]),
                                np.array([-s.x_start / 2, -1.0, 4.0]),
                                np.array([-s.x_start / 2, 1.0, -6.0]),
                                np.array([-s.x_start / 2, 1.0, -4.0]),
                                np.array([-s.x_start / 2, -1.0, -4.0])])
            self.r2 = np.array([np.array([-s.x_start, 0.0, 0.0]),
                                np.array([-s.x_start / 2, -1.0, 6.0]),
                                np.array([-s.x_start / 2, 1.0, 6.0]),
                                np.array([-s.x_start / 2, -1.0, 6.0]),
                                np.array([-s.x_start / 2, -1.0, -6.0]),
                                np.array([-s.x_start / 2, 1.0, -6.0]),
                                np.array([-s.x_start / 2, -1.0, -6.0])])
            self.flag_grab = np.array([False, True, False, False, True, False, False])

        def get_handrails(x: float, sequence: list):
            return [np.array([-x, sequence[0][k], sequence[1][k]]) for k in range(3)] + \
                [np.array([-x, sequence[0][k], -sequence[1][k]]) for k in range(3)] + \
                [np.array([-x, -sequence[1][k], sequence[0][k]]) for k in range(3)] + \
                [np.array([-x, sequence[1][k], sequence[0][k]]) for k in range(3)]

        if choice in ['2', '3', '4']:
            r_container = s.h * s.lvl * 1.5
            self.n = 13
            self.id = list(range(self.n))
            self.mass = [50.] + [30.] * (self.n - 1)  # ШАМАНСТВО, убрать пару ноликов
            self.diam = [5.] + [0.1] * (self.n - 1)
            sequence_1 = [[1., 1., -1.], [6., 4., 4.]]
            sequence_2 = [[-1., 1., -1.], [6., 6., 6.]]
            self.r1 = [np.zeros(3)] + get_handrails(x=s.x_start / 2, sequence=sequence_1)
            self.r2 = [np.array([-s.x_start * 1, 0.0, 0.0])] + get_handrails(x=s.x_start / 2, sequence=sequence_2)  # ШАМАНСТВО
            self.flag_grab = [False, True, False, False, True, False, False, True, False, False, True, False, False]

            r_beams = 0.05
            r_around = r_container - 0.2
            self.r_around = r_around
            self.id += [self.n]
            self.mass += [0.5]
            self.diam += [r_around]
            self.r1 += [[-s.x_start, 0, 0]]
            self.r2 += [[-s.x_start - s.container_length, 0, 0]]
            self.flag_grab += [False]
            self.n += 1

            '''r_beams = 0.05
            r_around = r_container - 0.2
            n_around = round(2 * np.pi * r_container / r_beams * 0.6)
            self.r_around = r_around
            for i in range(n_around):
                self.id += [self.n]
                self.mass += [0.5]
                self.diam += [r_beams]
                angle = i / n_around * 2 * np.pi
                self.r1 += [[-s.x_start, r_around * np.cos(angle), r_around * np.sin(angle)]]
                self.r2 += [[-s.x_start - s.container_length, r_around * np.cos(angle), r_around * np.sin(angle)]]
                self.flag_grab += [False]
                self.n += 1

            n_crossbar = int(s.lvl * 2)
            rotation_matrix = np.array([[1., 0., 0.], [0., -1 / 2, -np.sqrt(3) / 2], [0., np.sqrt(3) / 2, -1 / 2]])
            for i in range(3 * n_crossbar):
                self.id += [self.id[self.n - 1]]
                self.mass += [0.5]
                self.diam += [r_beams]
                self.flag_grab += [False]
                self.n += 1
            r1 = [np.array([-s.x_start - s.container_length * 0.92, 0., 0.]) for _ in range(n_crossbar)]
            r2 = [np.array([-s.x_start - s.container_length * 0.92, 0., 0.]) for _ in range(n_crossbar)]
            for i in range(s.lvl):
                r1[i][1] = s.h * (i + 1 / 2)
                r1[i][2] = np.sqrt(r_around ** 2 - r1[i][1] ** 2)
                r2[i][1] = s.h * (i + 1 / 2)
                r2[i][2] = -np.sqrt(r_around ** 2 - r2[i][1] ** 2)
                r1[i + s.lvl][1] = - s.h * (i + 1 / 2)
                r1[i + s.lvl][2] = np.sqrt(r_around ** 2 - r1[i + s.lvl][1] ** 2)
                r2[i + s.lvl][1] = - s.h * (i + 1 / 2)
                r2[i + s.lvl][2] = -np.sqrt(r_around ** 2 - r2[i + s.lvl][1] ** 2)
            self.r1 += r1
            self.r2 += r2
            r1 = [rotation_matrix @ r1[i] for i in range(n_crossbar)]
            r2 = [rotation_matrix @ r2[i] for i in range(n_crossbar)]
            self.r1 += r1
            self.r2 += r2
            r1 = [rotation_matrix @ r1[i] for i in range(n_crossbar)]
            r2 = [rotation_matrix @ r2[i] for i in range(n_crossbar)]
            self.r1 += r1
            self.r2 += r2'''

            self.id = np.array(self.id)
            self.mass = np.array(self.mass)
            self.diam = np.array(self.diam)
            self.r1 = np.array(self.r1)
            self.r2 = np.array(self.r2)
            self.flag_grab = np.array(self.flag_grab)

        if choice == '5':
            direction = ['+2', '-0', '+1', '+0', '-2', '-0', '-1', '-2', '+0', '-1', '+0', '+1', '+2', '-1', '+2',
                         '-0', '-2', '-0', '+2', '+2', '+1', '+0', '+1', '+0', '-1', '-2', '+1', '-2', '+1', '-0',
                         '-2', '-1', '-0', '+1', '+2', '-0', '-1', '+2', '+1', '+0', '+2', '+0', '-2', '+0', '+2',
                         '+0', '-1', '-2', '+1', '+0', '-1', '-2', '-0', '-1', '+2', '-1', '+2', '+1', '+0', '-2',
                         '-1', '-2', '-0', '-2', '+0', '+1', '-0', '+1', '+1', '+2', '+0', '-2', '-1', '+0', '-1',
                         '+2', '+1', '+2', '+1', '+2', '-0', '-1', '+0', '-1', '-2', '-1', '-2', '+0', '+1', '+2',
                         '-1', '+2', '-0']
            self.n = len(direction) + 1
            self.id = np.arange(self.n)
            self.mass = np.array([50.] * self.n)
            self.diam = np.array([0.5] * (self.n - 1) + [0.05])
            r1 = []
            r2 = []
            point = np.array([0., 0., 0.])
            for i in range(len(direction)):
                r1 += [point.copy()]
                point[int(direction[i][1])] += 4. if direction[i][0] == '+' else -4.
                r2 += [point.copy()]
            self.r1 = np.array(r1 + [r2[len(r2) - 1]])
            self.r2 = np.array(r2 + [r2[len(r2) - 1] + [0., 0., 1.]])
            self.flag_grab = np.array([False] * (self.n - 1) + [True])

        self.n = len(self.mass)

    def copy(self):
        c = Container(choice=self.choice, s=self.s)
        c.n = self.n
        c.id = copy.deepcopy(self.id)
        c.mass = copy.deepcopy(self.mass)
        c.diam = copy.deepcopy(self.diam)
        c.r1 = copy.deepcopy(self.r1)
        c.r2 = copy.deepcopy(self.r2)
        c.flag_grab = copy.deepcopy(self.flag_grab)
        return c


class Apparatus(object):
    def __init__(self, X: Structure, n: int = 1, mass: float = 20.):  # mass_check
        if n > len(X.id):
            raise ValueError('Слишком много аппаратов! Получи ошибку!')
        self.X = X
        self.n = n
        self.id = np.arange(n)
        self.mass = np.array([mass] * n)
        self.flag_fly = np.array([False] * n)
        self.flag_start = np.array([True] * n)
        self.flag_beam = np.array([None] * n)
        self.flag_hkw = np.array([True] * n)
        self.flag_to_mid = np.array([True] * n)
        self.busy_time = np.array([(i + 1) * 100. for i in range(n)])
        self.v = np.array([np.zeros(3) for _ in range(n)])
        id_list = X.call_possible_transport([]) if n > 0 else [0]
        self.target = np.array([(np.array(X.r_st[id_list[i]]) + np.array([- X.length[id_list[i]], 0, 0]))
                                for i in range(n)])
        self.target_p = copy.deepcopy(self.target)
        self.r = copy.deepcopy(self.target)
        self.r_0 = np.array([100. for _ in range(n)])

    def copy(self):
        a = Apparatus(X=self.X, n=self.n, mass=self.mass[0])
        a.flag_fly = copy.deepcopy(self.flag_fly)
        a.flag_start = copy.deepcopy(self.flag_start)
        a.flag_beam = copy.deepcopy(self.flag_beam)
        a.flag_hkw = copy.deepcopy(self.flag_hkw)
        a.flag_to_mid = copy.deepcopy(self.flag_to_mid)
        a.busy_time = copy.deepcopy(self.busy_time)
        a.target_p = copy.deepcopy(self.target_p)
        a.target = copy.deepcopy(self.target)
        a.v = copy.deepcopy(self.v)
        a.r = copy.deepcopy(self.r)
        a.r_0 = copy.deepcopy(self.r_0)
        return a


def get_all_components(choice: str = '1', testing: bool = False, complete: bool = False, n_app: int = 1,
                       floor: int = 5, extrafloor: int = 0):
    """Функция инициализирует классы конструкции и аппаратов"""
    s = Structure(choice=choice, testing=testing, complete=complete, floor=floor, extrafloor=extrafloor)
    c = Container(s=s, choice=choice)
    a = Apparatus(s, n=n_app)
    return s, c, a


if __name__ == "__main__":
    _, _, _ = get_all_components(choice='4', testing=True)
