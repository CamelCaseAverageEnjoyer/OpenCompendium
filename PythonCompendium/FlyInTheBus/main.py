import matplotlib.pyplot as plt
import numpy as np

def clip(a: any, bot: float, top: float):
    res = np.array(a)
    if np.linalg.norm(res) > top:
        return res / np.linalg.norm(res) * top
    if np.linalg.norm(res) < bot:
        return res / np.linalg.norm(res) * bot
    return res

class Bus:
    """Автобус"""
    def __init__(self, r_0: any = np.zeros(3), dims: any = (6.5, 1.5, 2.), control: str = "fixed", dt_: float = 0.1,
                 velocity: float = 100):
        self.r = np.array(r_0)
        self.v = np.zeros(3)
        self.dims = dims
        self.control = control
        self.dt = dt_
        self.velocity = velocity / 3.6

        self.t = 0.
        self.i = 0

    def act(self):
        self.i += 1
        self.t = self.i * self.dt
        if self.control == "fixed":
            pass
        if self.control == "uniform":
            self.v = np.array([self.velocity, 0, 0])
        if self.control == "sin":
            self.v = np.array([self.velocity, 0, 0]) * (1 + np.cos(self.t / 20) / 2)
        if self.control == "acceleration":
            self.v = np.array([self.velocity, 0, 0]) * self.t / 100

        self.r += self.v * self.dt


class Fly:
    """Материальная точка в роли мухи"""
    def __init__(self, b_: Bus, r_0: any, v_0: any, control: str = "pd", max_v: float = 1, max_a: float = 10,
                 random_weight: float = 0.86, k_pd: float = 0.03, k_bd: float = 10, points: int = 10):
        self.b = b_
        self.dt = b_.dt
        self.r = np.array(r_0)
        self.v = np.array(v_0)
        self.control = control
        self.max_a = max_a
        self.max_v = max_v
        self.random_weight = random_weight
        self.k_pd = k_pd
        self.k_bd = k_bd
        self.points = points

        self.a = np.zeros(3)
        self.a_a = np.zeros(3)
        self.x_line = [self.r[0]] * points
        self.y_line = [self.r[1]] * points
        self.z_line = [self.r[2]] * points
        self.x_scatter = []
        self.y_scatter = []
        self.z_scatter = []
        self.counter = 0
        self.save_rate = 100

    def act(self):
        self.b.act()

        self.a_a += np.random.uniform(-self.max_a, self.max_a, 3) * 300  # * self.dt
        self.a_a = clip(self.a_a, 0, self.max_a)
        self.a = np.zeros(3)
        if self.control == "no":
            pass
        if "pd" in self.control.split("+"):
            self.a += - self.k_pd * (self.r - self.b.r) - 2 * np.sqrt(self.k_pd) * (self.v - self.b.v)
        if "bd" in self.control.split("+"):
            for i in range(3):
                for j in [-1, 1]:
                    self.a[i] += self.k_bd / (self.r[i] - j * self.b.dims[i] - self.b.r[i]) ** 3

        self.v += clip(self.a_a * self.random_weight + self.a * (1 - self.random_weight), 0, self.max_a) * self.dt
        # self.v = self.b.v + clip(self.v - self.b.v, 0, self.max_v)
        self.v = self.b.v + clip(self.v - self.b.v, 0, self.max_v)
        self.r += self.v * self.dt

        # Муха бъётся о стекло
        for i in range(3):
            if abs(self.r[i] - self.b.r[i]) > self.b.dims[i]:
                self.x_scatter += [self.r[0] - self.b.r[0]]
                self.y_scatter += [self.r[1] - self.b.r[1]]
                self.z_scatter += [self.r[2] - self.b.r[2]]
                self.v[i] = -(self.v[i] - self.b.v[i]) + self.b.v[i]

    def show(self, is_return: bool = False):
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self.x_line + self.b.r[0], self.y_line + self.b.r[1], self.z_line + self.b.r[2])
        ax.scatter(self.x_scatter + self.b.r[0], self.y_scatter + self.b.r[1], self.z_scatter + self.b.r[2], c='r')

        # Отрисовка автобуса
        for i in [[-1, 1, -1, -1, -1, -1], [-1, 1, 1, 1, 1, 1],
                  [-1, -1, -1, 1, -1, -1], [1, 1, -1, 1, 1, 1],
                  [-1, -1, -1, -1, -1, 1], [1, 1, 1, 1, -1, 1],

                  [-1, 1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, 1],
                  [-1, -1, -1, 1, 1, 1], [1, 1, -1, 1, 1, 1],
                  [1, 1, -1, -1, -1, 1], [1, 1, 1, 1, -1, 1],

                  [-1, 1, -1, -1, 1, 1], [-1, 1, 1, 1, 1, 1],
                  [1, 1, -1, 1, -1, -1], [1, 1, -1, 1, 1, 1],
                  [-1, -1, 1, 1, -1, 1], [1, 1, 1, 1, -1, 1]]:
            ax.plot([i[0]*self.b.dims[0] + self.b.r[0], i[1]*self.b.dims[0] + self.b.r[0]],
                    [i[2]*self.b.dims[1] + self.b.r[1], i[3]*self.b.dims[1] + self.b.r[1]],
                    [i[4]*self.b.dims[2] + self.b.r[2], i[5]*self.b.dims[2] + self.b.r[2]], 'violet')

        ax.set_aspect('equal')
        if is_return:
            return ax
        else:
            plt.show()

    def integrate(self, t_: float, save_anim: bool = False):
        for _ in range(int(t_ // self.dt)):
            self.act()
            self.x_line += [self.r[0] - self.b.r[0]]
            self.y_line += [self.r[1] - self.b.r[1]]
            self.z_line += [self.r[2] - self.b.r[2]]
            '''self.x_line += [self.r[0]]
            self.y_line += [self.r[1]]
            self.z_line += [self.r[2]]'''

            if save_anim and self.b.i % self.save_rate == 0:
                self.counter += 1
                ax = self.show(is_return=True)
                plt.savefig(f"res/result_{self.counter:03}.jpg")
                plt.close()


if __name__ == '__main__':
    dt = 0.01
    t = 1000

    b = Bus(control=["fixed", "uniform", "sin", "acceleration"][2], dt_=dt)
    f = Fly(b_=b, r_0=np.zeros(3), v_0=np.random.uniform(-0.5, 0.5, 3), control=["no", "pd", "bd", "pd+bd"][2])
    f.save_rate = 300

    f.integrate(t, save_anim=True)
    f.show()
