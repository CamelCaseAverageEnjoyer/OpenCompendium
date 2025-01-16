# ---> в _tiny_functions.py

function [r, v, D] = orbitalMotionKeplerian(mu, p, epsilon, phi, omega, inc, t_pi, t_cur, approx)
%ORBITALMOTIONKEPLERIAN returns position and velocity of the satellite
%   mu - gravitational parameter
%   p - focal parameter
%   epslon - excentricity
%   phi - longitude of the ascending node
%   omega - argument of the pericenter
%   inc - inclination
%   t_pi - pericenter time
%   t_cur - current time
%   approx - first step for Newton Method

if epsilon < 1
    a = p/(1 - epsilon^2);
    b = sqrt(a*p);
    M = sqrt(mu/a^3)*(t_cur - t_pi);
    misclosure = 1;
    D = approx;
    while misclosure > 1e-11
        D = (epsilon*sin(D) - epsilon*cos(D)*D + M)/(1 - epsilon*cos(D));
%         D = D - (D - epsilon*sin(D) - M)/(1 - epsilon*cos(D));
        misclosure = abs(D - epsilon*sin(D) - M);
    end
%     display(D - epsilon*sin(D) - M)
    r1 = [a*(cos(D) - epsilon); b*sin(D); 0];
    cosTheta = (cos(D) - epsilon)/(1 - epsilon*cos(D));
    sinTheta = sqrt(1 - epsilon^2)*sin(D)/(1 - epsilon*cos(D));
    Vr = sqrt(mu/p)*epsilon*sinTheta;
    Vn = sqrt(mu/p)*(1 + epsilon*cosTheta);
    v1 = [Vr*cosTheta - Vn*sinTheta; Vr*sinTheta + Vn*cosTheta; 0];
else
    display('nonelliptic!')
    r = -1;
    v = -1;
    D = -1;
    return;
end
A1 = [cos(phi), sin(phi), 0;...
     -sin(phi), cos(phi), 0;...
        0,        0,    1];
    
A2 = [1,      0,          0;...
      0,  cos(inc), sin(inc);...
      0, -sin(inc), cos(inc)];
  
A3 = [cos(omega), sin(omega), 0;...
     -sin(omega), cos(omega), 0;... 
           0,         0,      1];
       
B = (A1')*(A2')*(A3');

r = B*r1;
v = B*v1;
end

