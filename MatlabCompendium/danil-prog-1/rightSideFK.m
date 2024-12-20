function [dx]=rightSideFK(x, w, u)

% HCW equations with control

dx(1:3) = x(4:6);

dx(4) = -2*w*x(6)+u(1);
dx(5) = -w^2*x(2)+u(2);
dx(6) = 2*w*x(4)+3*w^2*x(3)+u(3);
dx = dx';