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
#################################################

N_heatmap = 50
rate = 10
dx = 2
x_list = np.linspace(-dx, dx, N_heatmap)
y_list = np.linspace(-dx, dx, N_heatmap)

d = 0.5
dr = np.array([0.1, 0.1, 0])
ro1, ro2 = np.array([-1, 0, 0]), np.array([1, 0, 0])

def point2line(p, r1, r2):
    tau = (r2 - r1) / norm(r2 - r1)
    kk = np.dot(tau, p - r1)  # Положение проекции точки P на линии отрезка
    if 0<=kk<=norm(r2 - r1):  # Если проекция внутри отрезка
        return norm(p - (r1 + kk*tau))
    return -1

def point2circle(p, r1, r2):
    n = (r2 - r1) / norm(r2 - r1)
    d1 = norm(p - r2)
    cosa = np.dot(n, p - r2) / d1
    sina = np.sqrt(1 - cosa**2)
    kk = d1 * sina
    if cosa >= 0:
        if 0<=kk<=d:
            return d1*cosa
        if kk>d:
            return np.sqrt(((d1)*cosa)**2 + ((d1-d)*sina)**2)
    return -1

def h(r1, r2, ro1, ro2):
    # Проверка расстояний от точек до торцов цилиндра
    d1 = [point2circle(r1, ro1, ro2), 
          point2circle(r2, ro1, ro2), 
          point2circle(r1, ro2, ro1),
          point2circle(r2, ro2, ro1),
          point2circle(r1/2+r2/2, ro2, ro1),
          point2circle(r1/2+r2/2, ro2, ro1)]
    tmp = [i for i in d1 if i != -1]
    anw = -1 if len(tmp) == 0 else min(tmp)
    
    # Проверка расстояний между отрезками
    d2 = [point2line(r1, ro1, ro2) - d, 
          point2line(r2, ro1, ro2) - d, 
          point2line(ro1, r1, r2) - d,
          point2line(ro2, r1, r2) - d
         ]
    tmp = [i for i in d2 if i > -1]
    tmp = -1 if len(tmp) == 0 else min(tmp)
    # anw = min(tmp, anw) if anw >=0 else tmp
    if anw == -1 and tmp == -1:
        pass
    if anw == -1 and tmp != -1:
        anw = tmp
    if anw != -1 and tmp == -1:
        anw = anw
    if anw != -1 and tmp != -1:
        anw = min(anw, tmp)

    return np.clip(anw, 0, None)

def k(x_):
    return (x_ / (2*dx) + 0.5) * N_heatmap
def k1(x_):
    return (x_ / (2*dx)) * N_heatmap
def k_(i_):
    return (i_ / N_heatmap - 0.5) * (2*dx)
anw = [[h(r1=np.array([k_(ix), k_(iy), 0]) - dr, 
          r2=np.array([k_(ix), k_(iy), 0]) + dr,
          ro1=ro1, ro2=ro2) for ix in range(N_heatmap)] for iy in range(N_heatmap)]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
v_x_list, v_z_list = [np.linspace(-dx, dx, N_heatmap) for _ in range(2)]
xlabels_0, ylabels_0 = [['{:4.2f}'.format(x) for x in l] for l in [v_z_list, v_x_list]]
ax = sns.heatmap(anw, ax=axes, cmap="crest", xticklabels=xlabels_0, yticklabels=ylabels_0,  cbar_kws={'label': 'Distance h, m'})
ax.set_xticks(ax.get_xticks()[::rate]); ax.set_xticklabels(xlabels_0[::rate])
ax.set_yticks(ax.get_yticks()[::rate]); ax.set_yticklabels(ylabels_0[::rate])
ax.plot([k(ro1[0]), k(ro2[0])], [k(ro1[1]+d), k(ro2[1]+d)], c='c')
ax.plot([k(ro1[0]), k(ro2[0])], [k(ro1[1]-d), k(ro2[1]-d)], c='c')
ax.plot([k(ro1[0]), k(ro1[0])], [k(ro1[1]-d), k(ro1[1]+d)], c='c')
ax.plot([k(ro2[0]), k(ro2[0])], [k(ro2[1]-d), k(ro2[1]+d)], c='c')
ax.plot([k(ro1[0]), k(ro2[0])], [k(ro1[1]), k(ro2[1])], c='b')
for ix in range(N_heatmap):
    for iy in range(N_heatmap):
        if ((ix+1) % 5 == 0) and ((iy+1) % 5 == 0):
            ax.plot([ix-k1(dr[0]), ix+k1(dr[0])], [iy-k1(dr[1]), iy+k1(dr[1])], c='k')
            # ax.plot([ix-(dr[0]), ix+(dr[0])], [iy-(dr[1]), iy+(dr[1])], c='k')
ax.set_xlabel(f"x, m")
ax.set_ylabel(f"y, m")
ax.invert_yaxis()
plt.show()
