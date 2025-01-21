import numpy as np
from qdot import qdot
from q2dcm import q2dcm
from get_cube import *

class Dynamics():
    def __init__(self, h_orb, dt):
        self.r_earth = 6371e3
        self.R_sun = np.array([0, -150e6, 0])
        self.mu = 5.972e24 * 6.67408e-11  # standart gravitational parameter
        self.t = 0
        
        self.h_orb = h_orb
        self.r_orb = self.r_earth + self.h_orb
        self.w_orb = np.sqrt(self.mu / self.r_orb**3)
        self.U, self.R_orb = self.get_transition()
        self.dt = dt

    def get_transition(self):
        w = self.t * self.w_orb
        U = np.array([[np.cos(w), np.sin(w), 0],
                      [-np.sin(w), np.cos(w), 0],
                      [0, 0, 1]])
        R_orb = U.T @ np.array([self.r_orb, 0, 0])
        return U, R_orb

    def i2o_r(self, r_irf):
        return self.U @ r_irf - np.array([self.r_orb, 0, 0])
    def o2i_r(self, r_orf):
        return self.U.T @ (r_orf + [self.r_orb, 0, 0])
    def i2o_w(self, w_irf):
        return self.U @ (w_irf - [0, 0, self.w_orb])
    def o2i_w(self, w_orf):
        return self.U.T @ w_orf + [0, 0, self.w_orb]

    def rhs(self, r_orf, v_orf, q_irf, w_irf):
        dr = v_orf
        dv = np.array([3 * self.w_orb**2 * r_orf[0] + 2 * self.w_orb * v_orf[1],
                       -2 * self.w_orb * v_orf[0],
                       -self.w_orb**2 * r_orf[2]])
        dq = qdot(q_irf, [0] + list(w_irf)) / 2
        dw = np.zeros(3)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return dr, dv, dq, dw

    def rk4_integrate(self, obj):
        r = obj.r_orf
        v = obj.v_orf
        q = obj.q_irf
        w = obj.w_irf
        k1r, k1v, k1q, k1w = self.rhs(r, v, q, w);
        k2r, k2v, k2q, k2w = self.rhs(r+k1r*self.dt/2, v+k1v*self.dt/2, q+k1q*self.dt/2, w+k1w*self.dt/2);
        k3r, k3v, k3q, k3w = self.rhs(r+k2r*self.dt/2, v+k2v*self.dt/2, q+k2q*self.dt/2, w+k2w*self.dt/2);
        k4r, k4v, k4q, k4w = self.rhs(r+k3r*self.dt, v+k3v*self.dt, q+k3q*self.dt, w+k3w*self.dt);
        r = r + (k1r + 2*k2r + 2*k3r + k4r) * self.dt / 6
        v = v + (k1v + 2*k2v + 2*k3v + k4v) * self.dt / 6
        q = q + (k1q + 2*k2q + 2*k3q + k4q) * self.dt / 6
        w = w + (k1w + 2*k2w + 2*k3w + k4w) * self.dt / 6
        q = q / np.linalg.norm(q);
        return r, v, q, w

    def time_step(self, objs):
        self.t = self.t + self.dt
        self.U, self.R_orb = self.get_transition()

        for obj in objs:
            obj.r_orf, obj.v_orf, obj.q_irf, obj.w_irf = self.rk4_integrate(obj)
            obj.r_irf = self.o2i_r(obj.r_orf);
            obj.w_orf = self.i2o_w(obj.w_irf);

class Spacecraft():
    def __init__(self, d, r_orf, v_orf, q_irf, w_orf):
        cam_pos = None
        cam_dir = None
        cam_up = None
        self.v_orf = v_orf
        self.r_orf = r_orf
        self.r_irf = d.o2i_r(r_orf)
        self.q_irf = q_irf
        self.w_orf = w_orf
        self.w_irf = d.o2i_w(w_orf)

    def get_campos_irf(self):
        A = q2dcm(self.q_irf)
        return A @ self.cam_pos + self.r_irf
    def get_camdir_irf(self):
        A = q2dcm(self.q_irf)
        return A @ self.cam_dir
    def get_camup_irf(self):
        A = q2dcm(self.q_irf)
        return A @ self.cam_up

    def get_campos_orf(self, d):
        A = q2dcm(self.q_irf)
        S = A @ d.U.T 
        return S @ self.cam_pos + self.r_orf
    def get_camdir_orf(self, d):
        A = q2dcm(self.q_irf)
        S = A @ d.U.T 
        return S @ self.cam_dir
    def get_camup_orf(self, d):
        A = q2dcm(self.q_irf)
        S = A @ d.U.T 
        return S @ self.cam_up

    def show_chief(self, d):
        dims = np.array([0.3, 0.5, 0.3])
        A = q2dcm(self.q_irf)
        return get_cube(dims, A @ d.U.T, self.r_orf)

    def show_deputy(self, d):
        # from show_aruco import show_aruco

        # Corpus show (box)
        dims = np.array([0.1, 0.1, 0.01])
        A = q2dcm(self.q_irf)
        S = A @ d.U.T 
        mesh1 = get_cube(dims, A @ d.U.T, self.r_orf)   

        # Aruco show
        dim1 = 0.03
        r_brf = np.array([dims[0]/2 - dim1/2,
                          dims[1]/2 - dim1/2, 
                          dims[2]/2 + 0.003])
        # show_marker(r, [d(1)/2-d1/2; d(2)/2-d1/2; d(3)/2+0.0001], q, d1);  
        return mesh1  # + show_aruco(dim1, A, self.r_irf, r_brf)


