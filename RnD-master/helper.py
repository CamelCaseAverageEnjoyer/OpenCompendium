import numpy as np
import time
import vedo
from vedo.pyplot import plot

def func(event):
    t = time.time() - t0
    x = np.linspace(t, t + 4*np.pi, 100)
    p = plot(x, np.sin(x), ylim=(-1.2, 1.2))
    p.shift(-x[0])
    # pop (remove) the old and add the new one
    plt.pop().add(p).reset_camera()


t0 = time.time()
plt = vedo.Plotter()
plt.add_callback("timer", func)
plt.timer_callback("create", dt=10)
plt.show()
