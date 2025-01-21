classdef dynamics
    properties
        r_orb  % scalar value
        R_orb  % vector value
        w_orb {mustBePositive}  % orbital rate
        U  % rotation IRF->ORF
        r_earth = 6371 * 1000
        R_sun = [0;-15000000;15000000000]
        mu = 5.972e24 * 6.67408e-11  % standart gravitational parameter
        t = 0
        dt
    end

    methods
        function self = dynamics(h_orb, dt)
            self.r_orb = self.r_earth + h_orb;
            self.w_orb = sqrt(self.mu / self.r_orb^3);
            [self.U, self.R_orb] = self.get_transition();
            self.dt = dt;
        end

        function [U, R_orb] = get_transition(self)
            w = self.t * self.w_orb;
            U = [cos(w), sin(w), 0;
                 -sin(w), cos(w), 0;
                 0, 0, 1];
            R_orb = U.' * [self.r_orb; 0; 0];
        end

        function r_orf = i2o_r(self, r_irf)
            r_orf = self.U * r_irf - [self.r_orb; 0; 0];
        end
        function r_irf = o2i_r(self, r_orf)
            r_irf = self.U.' * (r_orf + [self.r_orb; 0; 0]);
        end
        function w_orf = i2o_w(self, w_irf)
            w_orf = self.U * (w_irf - [0; 0; self.w_orb]);
        end
        function w_irf = o2i_w(self, w_orf)
            w_irf = self.U.' * w_orf + [0; 0; self.w_orb];
        end

        function [dr, dv, dq, dw] = rhs(self, r_orf, v_orf, q_irf, w_irf)
            dr = v_orf;
            dv = [3 * self.w_orb^2 * r_orf(1) + 2 * self.w_orb * v_orf(2);
                  -2 * self.w_orb * v_orf(1);
                  -self.w_orb^2 * r_orf(3)];
            dq = qdot(q_irf, [0;w_irf]) / 2;
            dw = [0;0;0];  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        end

        function [r, v, q, w] = rk4_integrate(self, obj)
            r = obj.r_orf;
            v = obj.v_orf;
            q = obj.q_irf;
            w = obj.w_irf;
            [k1r, k1v, k1q, k1w] = self.rhs(r, v, q, w);
            [k2r, k2v, k2q, k2w] = self.rhs(r+k1r*self.dt/2, v+k1v*self.dt/2, ...
                          q+k1q*self.dt/2, w+k1w*self.dt/2);
            [k3r, k3v, k3q, k3w] = self.rhs(r+k2r*self.dt/2, v+k2v*self.dt/2, ...
                          q+k2q*self.dt/2, w+k2w*self.dt/2);
            [k4r, k4v, k4q, k4w] = self.rhs(r+k3r*self.dt, v+k3v*self.dt, ...
                          q+k3q*self.dt, w+k3w*self.dt);
            r = r + (k1r + 2*k2r + 2*k3r + k4r) * self.dt / 6;
            v = v + (k1v + 2*k2v + 2*k3v + k4v) * self.dt / 6;
            q = q + (k1q + 2*k2q + 2*k3q + k4q) * self.dt / 6;
            w = w + (k1w + 2*k2w + 2*k3w + k4w) * self.dt / 6;
            q = q / norm(q);
        end

        function [self, objs] = time_step(self, objs)
            self.t = self.t + self.dt;

            [self.U, self.R_orb] = self.get_transition();

            for i=1:length(objs)
                [objs(i).r_orf, objs(i).v_orf, objs(i).q_irf, objs(i).w_irf]...
                    = self.rk4_integrate(objs(i));
                objs(i).r_irf = self.o2i_r(objs(i).r_orf);
                objs(i).w_orf = self.i2o_w(objs(i).w_irf);
            end
        end    
    end
end
