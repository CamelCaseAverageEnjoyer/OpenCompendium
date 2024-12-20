function [S] = grin_midnight(jd)
%GRIN_MIDNIGHT Summary of this function goes here
%   Detailed explanation goes here

T=floor(jd)+0.5;
T0=juliandate([2000 1 1 12 0 0]);
T=(T-T0)/36525;
S=100.4606184+36000.77005361*T+0.00038793*T^2-2.6*10^(-8)*T^3;
S=S*pi/180;

end

