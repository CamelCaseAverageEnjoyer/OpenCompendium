import sys
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QApplication, QLineEdit, QCheckBox, \
     QComboBox, \
     QInputDialog, QFileDialog
from simulation import save_simulation_trajectories, load_simulation_trajectories, find_close_solution
from my_plot import *

ICON_SIZE = 50

class Window(QWidget):

    def __init__(self, o):
        super().__init__()
        self.o = o
        self.path = "../images/"
        self.n = 100
        self.wb = 50
        self.textboxes = {}
        self.checkboxes = {}
        self.comboboxes = {}
        self.choices = {}
        self.labels = {}
        self.images = {}
        self.name_type_func = []
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.grid.setSpacing(10)
        self.config_choose_n = 0

        self.initUI()

    def reset_all(self):
        self.o.reset(config_choose_n=self.config_choose_n)

    def initUI(self):
        # Очистка layout
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().deleteLater()
        self.name_type_func = [[['', '', None, (1, 1)] for _ in range(self.n)] for _ in range(self.n)]

        # Подгрузка параметров + инициализация
        self.reset_all()

        # Редактирование сетки
        self.buttons_and_labels()

        positions = [(i, j) for i in range(self.n) for j in range(self.n)]
        names = [self.name_type_func[i][j][0] for i in range(self.n) for j in range(self.n)]
        types = [self.name_type_func[i][j][1] for i in range(self.n) for j in range(self.n)]
        funcs = [self.name_type_func[i][j][2] for i in range(self.n) for j in range(self.n)]
        whs = [self.name_type_func[i][j][3] for i in range(self.n) for j in range(self.n)]
        for position, name, t, f, wh in zip(positions, names, types, funcs, whs):
            if name == '':
                continue
            if 'button' in t:
                if 'jpg' in name or 'png' in name:
                    button = QPushButton('')
                    button.setIcon(QIcon(name))
                    button.setIconSize(QSize(self.wb, self.wb))
                else:
                    button = QPushButton(name)
                if f is not None:
                    # button.clicked.connect(f)
                    button.pressed.connect(f)
                self.grid.addWidget(button, *position, *wh)
            elif 'label' in t:
                label = QLabel(name)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                label.setStyleSheet("border: 1px solid #C2C7CB; border-radius: 5%; padding: 10px;")
                self.labels[name] = label  # Задел на он-лайн изменение; потом
                self.grid.addWidget(label, *position, *wh)
            elif 'image' in t:
                label = QLabel()
                pix = QPixmap(name)
                label.setPixmap(pix)
                if ";" in t:
                    s = int(t.split(";")[1])
                    label.setPixmap(pix.scaled(s, s, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.FastTransformation))
                self.images[name] = label  # Задел на он-лайн изменение; потом
                self.grid.addWidget(label, *position, *wh)
            elif 'edit' in t:
                textbox = QLineEdit()
                textbox.setText(t.split(";")[1])
                self.textboxes[name] = textbox
                self.grid.addWidget(textbox, *position, *wh)
            elif 'check' in t:
                b = QCheckBox()
                b.setChecked(int(t.split(";")[1]) > 0)
                self.checkboxes[name] = b
                self.grid.addWidget(b, *position, *wh)
            elif 'combo' in t:
                c = QComboBox()
                c.addItems(f.split(";"))
                c.setCurrentText(t.split(";")[1])
                self.comboboxes[name] = c
                self.grid.addWidget(c, *position, *wh)

        # Редактирование окна
        self.move(400, 0)
        self.setWindowTitle('kiam-formation')
        self.setStyleSheet('background-color: grey;')
        self.setWindowIcon(QIcon(self.path + "wizard.png"))
        self.show()

    def buttons_and_labels(self):
        from cosmetic import talk

        data = self.o.v.config_choose
        params = data.iloc[self.config_choose_n, :]
        y_all = 0
        n = 0

        # >>>>>>>>>>>> Кнопки <<<<<<<<<<<<
        y = 0
        self.name_type_func[y][n] = [self.path + "robot1.png", "button", talk, (1, 1)]
        # self.name_type_func[y][n+1] = ["Помощь", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "integral.png", "button", self.main_run, (1, 1)]
        # self.name_type_func[y][n+1] = ["Численное\nмоделирование", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "plot.png", "button", lambda x=self.o: plot_distance(x), (1, 1)]
        # self.name_type_func[y][n+1] = ["Показать результаты\nзадачи навигации", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "orbit.png", "button", lambda x=self.o: plot_all(x), (1, 1)]
        # self.name_type_func[y][n+1] = ["Отобразить в 3D", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "param.png", "button", self.plot_1_param, (1, 1)]
        # self.name_type_func[y][n+1] = ["Выборочно отобразить\nзаписанные параметры", "label", None, (1, 1)]
        # y += 1
        # self.name_type_func[y][n] = [self.path + "path.png", "button", self.local_solve_minimization, (1, 1)]
        # self.name_type_func[y][n+1] = ["Навигация роя\nс помощью scipy", "label", None, (1, 1)]
        # y += 1
        # self.name_type_func[y][n] = [self.path + "wizard.png", "button", lambda x=self.o: find_close_solution(x), (1, 1)]
        # self.name_type_func[y][n+1] = ["Прикольные\nприколы", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "antenna.png", "button", lambda x=self.o: plot_model_gain(x), (1, 1)]
        # self.name_type_func[y][n+1] = ["Посмотреть диаграммы\nнаправленностей", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "air.png", "button", plot_atmosphere_models, (1, 1)]
        # self.name_type_func[y][n+1] = ["Посмотреть модели\nатмосферы", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "animation.png", "button", animate_reference_frames, (1, 1)]
        # self.name_type_func[y][n+1] = ["Анимировать вращение\nвокруг Земли", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "save.png", "button", self.local_save_trajectories, (1, 1)]
        # self.name_type_func[y][n+1] = ["Сохранить траектории", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "load.png", "button", self.local_load_trajectories, (1, 1)]
        # self.name_type_func[y][n+1] = ["Загрузить траектории", "label", None, (1, 1)]
        y += 1
        self.name_type_func[y][n] = [self.path + "eraser.png", "button", self.local_remove_trajectories, (1, 1)]
        # self.name_type_func[y][n+1] = ["Удалить траектории", "label", None, (1, 1)]
        y += 1
        n += 1  # 2
        n_left = n

        # >>>>>>>>>>>> Параметры численного моделирования <<<<<<<<<<<<

        # Первый столбец - выбор из вариантов
        y = 1
        self.name_type_func[y][n+0] = ["dT", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["dT", f"combo;{params['dT']}", ";".join(self.o.v.dTs), (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["T", "label", "", (1, 1)]
        self.name_type_func[y][n+1] = ["TIME", f"combo;{params['TIME']}", ";".join(self.o.v.Ts), (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["A-priori", "label", "", (1, 1)]
        self.name_type_func[y][n+1] = ["START_NAVIGATION_N",
                                       f"combo;{self.o.v.NAVIGATIONS[params['START_NAVIGATION_N']]}",
                                       ";".join(self.o.v.NAVIGATIONS), (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["CubeSat ant", "label", "", (1, 1)]
        self.name_type_func[y][n+1] = ["GAIN_MODEL_C_N",
                                       f"combo;{self.o.v.GAIN_MODES[params['GAIN_MODEL_C_N']]}",
                                       ";".join(self.o.v.GAIN_MODES), (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["ChipSat ant", "label", "", (1, 1)]
        self.name_type_func[y][n+1] = ["GAIN_MODEL_F_N",
                                       f"combo;{self.o.v.GAIN_MODES[params['GAIN_MODEL_F_N']]}",
                                       ";".join(self.o.v.GAIN_MODES), (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["CubeSat", "label", "", (1, 1)]
        self.name_type_func[y][n+1] = ["CUBESAT_MODEL_N",
                                       f"combo;{self.o.v.CUBESAT_MODELS[params['CUBESAT_MODEL_N']]}",
                                       ";".join(self.o.v.CUBESAT_MODELS), (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["ChipSat", "label", "", (1, 1)]
        self.name_type_func[y][n+1] = ["CHIPSAT_MODEL_N",
                                       f"combo;{self.o.v.CHIPSAT_MODELS[params['CHIPSAT_MODEL_N']]}",
                                       ";".join(self.o.v.CHIPSAT_MODELS), (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["Deploy", "label", "", (1, 1)]
        self.name_type_func[y][n+1] = ["DEPLOYMENT_N",
                                       f"combo;{self.o.v.DEPLOYMENTS[params['DEPLOYMENT_N']]}",
                                       ";".join(self.o.v.DEPLOYMENTS), (1, 1)]
        y += 1
        n += 2
        y_all = max(y_all, y)


        # Второй столбец - вручную вбиваемые значения
        y = 1
        self.name_type_func[y][n+0] = ["CubeSat", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = [self.path + f"{self.o.v.CUBESAT_MODELS[params['CUBESAT_MODEL_N']]}.png",
                                       f"image;{ICON_SIZE}", None, (1, 1)]
        self.name_type_func[y][n+2] = ["CUBESAT_AMOUNT", f"edit;{params['CUBESAT_AMOUNT']}", None, (1, 3)]
        y += 1
        self.name_type_func[y][n+0] = ["ChipSat", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = [self.path + f"chipsat.png", f"image;{ICON_SIZE}", None, (1, 1)]
        self.name_type_func[y][n+2] = ["CHIPSAT_AMOUNT", f"edit;{params['CHIPSAT_AMOUNT']}", None, (1, 3)]
        y += 1
        self.name_type_func[y][n+0] = ["q", "label", None, (1, 1)]
        q = [float(k) for k in params['q'].split()]
        for k in range(2):
            self.name_type_func[y][n+1+k] = [f"q{k}", f"edit;{q[k]}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["p", "label", None, (1, 1)]
        p = [float(k) for k in params['p'].split()]
        for k in range(4):
            self.name_type_func[y][n+1+k] = [f"p{k}", f"edit;{p[k]}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["r", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["r", f"edit;{params['r']}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["CubeSat [r,v,ω]", "label", None, (1, 1)]
        rvw_cubesat = [float(k) for k in params['RVW_CubeSat_SPREAD'].split()]
        for k in range(3):
            self.name_type_func[y][n+1+k] = [f"rvw_cubesat{k}", f"edit;{rvw_cubesat[k]}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["ChipSat [r,v,ω]", "label", None, (1, 1)]
        rvw_chipsat = [float(k) for k in params['RVW_ChipSat_SPREAD'].split()]
        for k in range(3):
            self.name_type_func[y][n+1+k] = [f"rvw_chipsat{k}", f"edit;{rvw_chipsat[k]}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["Distortion", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["DISTORTION", f"edit;{params['DISTORTION']}", None, (1, 1)]
        y += 1
        n += 5
        y_all = max(y_all, y)

        # Третий столбец - чекбоксы
        y = 1
        self.name_type_func[y][n+0] = ["Aero", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["DYNAMIC_MODEL_aero", f"check;{int(params['DYNAMIC_MODEL_aero'])}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["J₂", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["DYNAMIC_MODEL_j2", f"check;{int(params['DYNAMIC_MODEL_j2'])}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["[q, ω]", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["NAVIGATION_ANGLES", f"check;{int(params['NAVIGATION_ANGLES'])}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["Kalman", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["IF_NAVIGATION", f"check;{int(params['IF_NAVIGATION'])}", None, (1, 1)]
        y += 1
        self.name_type_func[y][n+0] = ["Show", "label", None, (1, 1)]
        self.name_type_func[y][n+1] = ["RELATIVE_SIDES", f"check;{int(self.o.v.RELATIVE_SIDES)}", None, (1, 1)]
        y += 1
        n += 2
        y_all = max(y_all, y)

        self.name_type_func[0][n_left] = ["Параметры численного моделирования", "label", None, (1, n - 1)]

        # >>>>>>>>>>>> Автоматическое заполнение сохранений <<<<<<<<<<<<
        y_save = len(self.o.v.config_choose)
        self.name_type_func[y_save + 1][n+0] = ["Параметры", "edit;Название", None, (1, 2)]
        self.name_type_func[y_save + 1][n+2] = ["Сохранить", "button", self.local_save_params, (1, 1)]
        self.name_type_func[y_save + 2][n+0] = ["Применить введённые параметры", "button", self.apply_params, (1, 3)]
        self.name_type_func[y_save + 3][n+0] = ["Сбросить моделирование", "button", self.reset_all, (1, 3)]

        for i in range(y_save):
            self.name_type_func[i+1][n+0] = [data.iloc[i, 0], "label", None, (1, 1)]
            self.name_type_func[i+1][n+1] = [f"Загрузить", "button", lambda j=i: self.local_load_params(i=j),
                                                   (1, 1)]
            self.name_type_func[i+1][n+2] = [f"Удалить", "button", lambda j=i: self.local_remove_params(i=j),
                                                   (1, 1)]

        # Текст о долготе полёта
        # self.name_type_func[y_all][n_left+3] = [self.o.time_message(params['TIME']), "label", None, (1, 4)]

    def apply_params(self):
        """Применение настроенных параметров. Должно быть согласовано с config.get_saving_params"""
        if self.textboxes['Параметры'].text() != "":
            self.o.v.DESCRIPTION = self.textboxes['Параметры'].text()

        self.o.v.dT = float(self.comboboxes['dT'].currentText())
        self.o.v.TIME = float(self.comboboxes['TIME'].currentText())
        self.o.v.START_NAVIGATION = self.comboboxes['START_NAVIGATION_N'].currentText()
        self.o.v.START_NAVIGATION_N = self.o.v.NAVIGATIONS.index(self.o.v.START_NAVIGATION)
        self.o.v.GAIN_MODEL_C = self.comboboxes['GAIN_MODEL_C_N'].currentText()
        self.o.v.GAIN_MODEL_C_N = self.o.v.GAIN_MODES.index(self.o.v.GAIN_MODEL_C)
        self.o.v.GAIN_MODEL_F = self.comboboxes['GAIN_MODEL_F_N'].currentText()
        self.o.v.GAIN_MODEL_F_N = self.o.v.GAIN_MODES.index(self.o.v.GAIN_MODEL_F)
        self.o.v.CUBESAT_MODEL = self.comboboxes['CUBESAT_MODEL_N'].currentText()
        self.o.v.CUBESAT_MODEL_N = self.o.v.CUBESAT_MODELS.index(self.o.v.CUBESAT_MODEL)
        self.o.v.CHIPSAT_MODEL = self.comboboxes['CHIPSAT_MODEL_N'].currentText()
        self.o.v.CHIPSAT_MODEL_N = self.o.v.CHIPSAT_MODELS.index(self.o.v.CHIPSAT_MODEL)
        self.o.v.DEPLOYMENT = self.comboboxes['DEPLOYMENT_N'].currentText()
        self.o.v.DEPLOYMENT_N = self.o.v.DEPLOYMENTS.index(self.o.v.DEPLOYMENT)

        self.o.v.DISTORTION = float(self.textboxes['DISTORTION'].text())
        self.o.v.CUBESAT_AMOUNT = int(self.textboxes['CUBESAT_AMOUNT'].text())
        self.o.v.CHIPSAT_AMOUNT = int(self.textboxes['CHIPSAT_AMOUNT'].text())
        self.o.v.KALMAN_COEF['q'] = [float(self.textboxes[f'q{k}'].text()) for k in range(2)]
        self.o.v.KALMAN_COEF['p'] = [float(self.textboxes[f'p{k}'].text()) for k in range(4)]
        self.o.v.KALMAN_COEF['r'] = float(self.textboxes['r'].text())
        self.o.v.RVW_CubeSat_SPREAD = [float(self.textboxes[f'rvw_cubesat{k}'].text()) for k in range(3)]
        self.o.v.RVW_ChipSat_SPREAD = [float(self.textboxes[f'rvw_chipsat{k}'].text()) for k in range(3)]

        self.o.v.DYNAMIC_MODEL['aero drag'] = self.checkboxes['DYNAMIC_MODEL_aero'].isChecked()
        self.o.v.DYNAMIC_MODEL['j2'] = self.checkboxes['DYNAMIC_MODEL_j2'].isChecked()
        self.o.v.NAVIGATION_ANGLES = self.checkboxes['NAVIGATION_ANGLES'].isChecked()
        self.o.v.IF_NAVIGATION = self.checkboxes['IF_NAVIGATION'].isChecked()
        self.o.v.RELATIVE_SIDES = self.checkboxes['RELATIVE_SIDES'].isChecked()

        my_print('Параметры применены!', color='c')

    def local_save_params(self):
        """Сохранение настроенных параметров"""
        self.apply_params()
        self.o.v.save_params()
        self.config_choose_n = len(self.o.v.config_choose) - 1
        self.close()
        self.initUI()

    def local_load_params(self, i: int):
        self.config_choose_n = i
        self.close()
        self.initUI()

    def local_remove_params(self, i: int):
        if len(self.o.v.config_choose) > 1:
            self.o.v.remove_params(i)
        if i <= self.config_choose_n:
            self.config_choose_n -= 1
        self.close()
        self.initUI()

    def local_save_trajectories(self):
        text, ok = QInputDialog.getText(self, 'Сохранение траектории', 'Введите название файла:')
        if ok:
            save_simulation_trajectories(o=self.o, text=f"{self.o.v.path_sources}trajectories/{text}")

    def local_load_trajectories(self):
        path = f"{self.o.v.path_sources}trajectories/"
        text = QFileDialog.getOpenFileName(self, 'Open file', path)[0]
        if text != "":
            load_simulation_trajectories(o=self.o, text=path + text.split(path)[1])

    def local_remove_trajectories(self):
        from os import listdir, remove
        items = sorted([s for s in listdir(f"{self.o.v.path_sources}trajectories")])
        if len(items) > 0:
            text, ok = QInputDialog.getItem(self, "Удаление", "Выберите траекторию для удаления", items, 0, False)
            if ok:
                remove(f"{self.o.v.path_sources}trajectories/{text}")

    def local_solve_minimization(self):  # Можно и без локальной фукнции
        from simulation import solve_minimization_new
        solve_minimization_new(o=self.o, config_choose_n=self.config_choose_n)

    def plot_1_param(self):
        d = self.o.p.record
        items = sorted(d.columns)

        # ручная сортировка
        items = [i for i in items if not ('orf' in i or 'irf' in i)] + \
                [i for i in items if 'orf' in i] + [i for i in items if 'irf' in i]

        y_s = []
        ok = True
        while ok:
            text, ok = QInputDialog.getItem(self, "Отображение", "Выберите насколько параметров", items, 0, False)
            y_s.append(text)
        for y in y_s[:-1]:  # Костыль - в конце добавляется items[0]
            plt.plot(d['t'].to_list(), d[y].to_list(), label=y)
        plt.legend()
        plt.grid()
        plt.show()

    def main_run(self):
        """Функция запуска численного моделирования, выводы результатов"""
        import numpy as np
        from cosmetic import talk

        if self.o.p.iter < 2:
            my_print(f"Повторная инициализация...", color='y', if_print=self.o.v.IF_ANY_PRINT)
            self.o.init_classes()
        self.o.integrate(t=self.o.v.TIME, animate=False)

        # Вывод результатов
        tmp = np.array(self.o.p.record[f'{self.o.f.name} KalmanPosError r {0}'].to_list())  # Для дочернего КА с id=0
        print(f"Математическое ожидание ошибки: {tmp.mean():.2f} м, Среднее отклонение ошибки: {tmp.std():.2f} м")
        if self.o.v.IF_NAVIGATION:
            plot_distance(self.o)  # Авто-показ графика оценки движения

        if self.o.v.IF_TALK:
            talk()

def interface_window(o):
    app = QApplication(sys.argv)
    window = Window(o=o)
    return app, window
