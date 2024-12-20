function [r, v] = orbitalMotionKeplerian_phi(mu, p, phi, inc, u)
%ORBITALMOTIONKEPLERIAN returns position and velocity of the satellite
%   mu - gravitational parameter
%   p - focal parameter
%   epsilon - excentricity
%   phi - longitude of the ascending node
%   omega - argument of the pericenter
%   inc - inclination
%   t_pi - pericenter time
%   t_cur - current time
%   approx - first step for Newton Method

r1 = [p; 0; 0];
v1 = [0; sqrt(mu/p); 0];
A1 = [cos(phi), sin(phi), 0;...
     -sin(phi), cos(phi), 0;...
        0,        0,    1];
    
A2 = [1,      0,          0;...
      0,  cos(inc), sin(inc);...
      0, -sin(inc), cos(inc)];
  
A3 = [cos(u), sin(u), 0;...
     -sin(u), cos(u), 0;... 
           0,         0,      1];
       
B = (A1')*(A2')*(A3');

r = B*r1;
v = B*v1;
end

