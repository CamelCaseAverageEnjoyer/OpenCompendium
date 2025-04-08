import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import norm
from sympy import *
from __init__ import *

o = kf.init()
o.v.ORBIT_RADIUS = 6371e3 + 500e3
o.v.W_ORB = np.sqrt(o.v.MU / o.v.ORBIT_RADIUS ** 3)
print(f"w0 = {o.v.W_ORB}")
print(f"Период орбиты: {(2*np.pi / o.v.W_ORB):.2f} сек")

def a_orb(w, x, y, z, vx, vy, vz):
    return kf.vec_type([-2*w*vy,
                        3*w**2*y + 2*w*vx,
                        -w**2*z])

# Скоропировано из "Решений уравнений"
def A(w, t):
    return kf.vec_type([[1, -6*w*t + 6*kf.sin(w*t), 0],
                        [0, 4 - 3*kf.cos(w*t), 0],
                        [0, 0, kf.cos(w*t)]], b=t)

def B(w, t):
    return kf.vec_type([[4*kf.sin(w*t) - 3*w*t, 2*kf.cos(w*t) - 2, 0],
                        [-2*kf.cos(w*t) + 2, kf.sin(w*t), 0],
                        [0, 0, kf.sin(w*t)]], b=t) / w

def C(w, t):
    return kf.vec_type([[0, -6*w + 6*w*kf.cos(w*t), 0],
                        [0, 3*w*kf.sin(w*t), 0],
                        [0, 0, -w*kf.sin(w*t)]], b=t)

def D(w, t):
    return kf.vec_type([[4*kf.cos(w*t) - 3, -2*kf.sin(w*t), 0],
                        [2*kf.sin(w*t), kf.cos(w*t), 0],
                        [0, 0, kf.cos(w*t)]], b=t)

def v0x(w, x, y, z):
    return -2*y*w

def v0req(w, t, r0, r1):
    return kf.inv(B(w,t)) @ (r1 - A(w,t) @ r0)

def mycolor(a, c1=[7, 255, 224], c2=[218, 17, 255]):
    return tuple([(c1[i] * (1-a) + c2[i]*a)/255 for i in range(3)])

w, t = var('w_0 t', real=True)
r0 = kf.get_vars(name='r^0', n=3, numb=False)
v0 = kf.get_vars(name='v^0', n=3, numb=False)
r1 = kf.get_vars(name='r^1', n=3, numb=False)
v1 = kf.get_vars(name='v^1', n=3, numb=False)
r = [Function('r_x'), Function('r_y'), Function('r_z')]
v = [Function('v_x'), Function('v_y'), Function('v_z')]
a = a_orb(w, r[0](t), r[1](t), r[2](t), v[0](t), v[1](t), v[2](t))


dt = 1
T = 5000
w0 = o.v.W_ORB
r0 = np.array([1, 0, 1])
r1 = np.array([-1, 0, 0])
v0 = v0req(w0, T, r0, r1)

z1 = lambda r, v, w: np.array([r[0] - 2*v[1]/w,
                               r[1] + v[0]/2/w,
                               0])
z3 = lambda r, v, w: np.array([-w*(3*r[1]*v[2]*w + r[2]*v[1]*w + 2*v[0]*v[2]),
                               2*w*(3*r[1]*r[2]*w**2 + 2*r[2]*v[0]*w - v[1]*v[2]),
                               2*w*(9*r[1]**2*w**2 + 12*r[1]*v[0]*w + 4*v[0]**2 + v[1]**2)])

fig = plt.figure(figsize=(6,6))
axes = fig.add_subplot(projection='3d')
rn = np.zeros((T // dt,3))
for i in range(T // dt):
    rn[i,:] = A(w0, dt*i) @ r0 + B(w0, dt*i) @ v0
axes.scatter(*r0, c='b', label='Start point')
axes.scatter(*r1, c='r', label='End point')
axes.plot(*[rn[:,i] for i in range(3)], c='k', label='Trajectory')
for j in range(T // dt):
    if j % 100 == 0:
        # bn1 = bn(r0, v0, w0, dt*j).T[0]
        # bn1 /= np.linalg.norm(bn1)
        # axes.plot(*[[rn[j,i], rn[j,i] + 0.5*bn1[i]] for i in range(3)], c='r', label=None)
        
        bn2 = z3(r0, v0, w0) # .T[0]
        bn2 /= np.linalg.norm(bn2)
        axes.plot(*[[rn[j,i], rn[j,i] + 0.5*bn2[i]] for i in range(3)], c='g')
axes.legend()
axes.set_xlabel("x, m")
axes.set_ylabel("y, m")
axes.set_zlabel("z, m")
axes.set_aspect('equal')
plt.show()
