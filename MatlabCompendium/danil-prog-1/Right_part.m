function [dX] = Right_part(t,X,f_a)
%RIGHT_PART Summary of this function goes here
%   Detailed explanation goes here
dX(1)=X(4);
dX(2)=X(5);
dX(3)=X(6);
Rad=norm(X(1:3));
mu=3.986*10^14;
% [gx gy gz]=gravitysphericalharmonic(X(1:3)');
R = 6378000;
delta = 3/2*1082.8e-6*mu*R^2;
x = X(1);
y = X(2); 
z = X(3);
r=[X(1);X(2);X(3)];
acceleration_J2 = delta*r/norm(r)^5*(5*z^2/norm(r)^2 - 1)- 2*delta/norm(r)^5*[0; 0; z];
% acceleration_J2 = [0, 0, 0];

dX(4)=-mu*X(1)/Rad^3+acceleration_J2(1)+f_a(1);
dX(5)=-mu*X(2)/Rad^3+acceleration_J2(2)+f_a(2);
dX(6)=-mu*X(3)/Rad^3+acceleration_J2(3)+f_a(3);
dX=dX';
end

