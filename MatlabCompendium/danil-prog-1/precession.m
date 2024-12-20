function [X, Y, Z] = precession(X, Y, Z, t)

Z(1,:) = -Z(2,:);
psi = 0.1*pi/180*t;
theta = 45*pi/180;
phi = 0.2*pi/180*t;
A = angle2dcm(psi, theta, phi,'ZXZ');
for j =1:1:2
    for i = 1:1:length(X(1,:))
       R = A*[X(j,i);Y(j,i);Z(j,i)];
       X(j,i)=R(1);
       Y(j,i)=R(2);
       Z(j,i)=R(3);
    end
end

end

