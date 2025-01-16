function [F] = dipole_force(mu_0,r,mu,Mu)
R = norm(r);
F = (3*mu_0/(4*pi*R^5))*(Mu'*mu*r + Mu'*r*mu + mu'*r*Mu - (5/R^2)*(Mu'*r)*(mu'*r)*r) ;
end