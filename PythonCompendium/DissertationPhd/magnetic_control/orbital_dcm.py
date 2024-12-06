function [ A ] = orbital_dcm(X)
%ORBITAL_DCM Summary of this function goes here
%   Detailed explanation goes here
z=X(1:3)/norm(X(1:3));
y=cross(X(1:3),X(4:6))/norm(cross(X(1:3),X(4:6)));
x=cross(y,z);

A=[x,y,z];
A=A';

end

