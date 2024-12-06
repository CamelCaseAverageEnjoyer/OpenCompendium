function [output] = Sedvick_param(parameters)
%SEDVICK_PARAM Summary of this function goes here
%   Detailed explanation goes here
J2=1.082*10^(-3);
mu=3.986*10^14;
R_Earth=6.378245*10^6;

r0=parameters(1);
i0=parameters(2);
i1=parameters(3);
i2=parameters(4);
z0=parameters(5);
dz0=parameters(6);

c=sqrt(1+3*J2*R_Earth^2*(1+3*cos(2*i0))/8/r0^2);
n=sqrt(mu/r0^3);
k=n*c+3*J2*R_Earth^2*cos(i0)^2/2/r0^2;
i2=dz0/k/r0+i1;

dW0=z0/r0/sin(i0);

q=n*c+3*n*J2*R_Earth^2/2/r0^2*(cos(i2)^2-((cos(i1)-cos(i2))*(tan(i1)^(-1)*sin(i2)*cos(dW0)-cos(i2)))/(sin(dW0)^2+(tan(i1)^(-1)*sin(i2)-cos(i2)*cos(dW0))^2));
l=-3*n*J2*R_Earth^2/2/r0*((cos(i1)-cos(i2))*sin(i1)*sin(i2)*sin(dW0))/sqrt(1-(cos(i1)*cos(i2)+sin(i1)*sin(i2)*cos(dW0))^2);

a=roots([l^2; -2*dz0*l; q^2*z0^2+dz0^2; 0; - q^2*z0^2]);
phi=asin(min(abs(a)));

output=[n c q l phi];

end

