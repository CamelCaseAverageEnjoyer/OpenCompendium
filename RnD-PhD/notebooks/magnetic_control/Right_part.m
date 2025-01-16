function [dX] = Right_part(t,X,Mu,B,J,F)
r = X(1:3);
v = X(4:6);
q = X(7:10);
w = X(11:13);

mu=3.986*10^14;
R = 6378000;
delta = 3/2*1082.8e-6*mu*R^2;
% acceleration_J2 = delta*r/norm(r)^5*(5*z^2/norm(r)^2 - 1)...
%     - 2*delta/norm(r)^5*[0; 0; z];
acceleration_J2 = [0; 0; 0];

dr = v;
dv = -mu*r/norm(r)^3 + acceleration_J2 + F;
dq = 0.5*quatmultiply(q', [0; w]')';

inv_J = inv(J);

q = quatnormalize(q');
A = quat2dcm(q); %to body frame
M_gr = 3*mu/norm(r)^5*cross(A*r, J*A*r); 
% M_gr = [0;0;0];

M_mag = cross(Mu, A*B); 
% M_mag = [0;0;0];

dw = inv_J*(-cross(w, J*w) + M_gr + M_mag);

dX = [dr; dv; dq; dw];

end