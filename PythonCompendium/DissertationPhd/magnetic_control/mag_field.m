function [B] = mag_field(mu_0,r,Mu) 
R = norm(r);
B = (mu_0/(4*pi*R^5))*(3*Mu'*r*r - R^2*Mu);
end 