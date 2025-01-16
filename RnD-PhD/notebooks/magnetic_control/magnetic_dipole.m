function [MU] = magnetic_dipole(F, d, mu1)
mu_0 = 4*pi*1e-7; % permeability of free space

syms mu2_x mu2_y mu2_z
eqn1 = (3*mu_0/(4*pi*(norm(d))^5))*((mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(1)+(mu1'*d)*mu2_x+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(1)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(1)) == F;
eqn2 = (mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(2)+(mu1'*d)*mu2_y+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(2)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(2) == 0;
eqn3 = (mu1(1)*mu2_x+mu1(2)*mu2_y+mu1(3)*mu2_z)*d(3)+(mu1'*d)*mu2_z+(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*mu1(3)-(5/(norm(d))^2)*(mu1'*d)*(mu2_x*d(1)+mu2_y*d(2)+mu2_z*d(3))*d(3) == 0;
sol = vpasolve([eqn1, eqn2, eqn3], [mu2_x mu2_y mu2_z]);
MU(1,1) = sol.mu2_x;
MU(2,1) = sol.mu2_y;
MU(3,1) = sol.mu2_z;
end