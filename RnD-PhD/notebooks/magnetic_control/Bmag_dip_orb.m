function [B, dB] = Bmag_dip_orb(t,u)

global incl B0 w02

%โ ฮัส
Bdip(3)=-2*sin(incl)*sin(u);
Bdip(1)=sin(incl)*cos(u);
Bdip(2)=cos(incl);

dBdip(3)=-2*w02*sin(incl)*cos(u);
dBdip(1)=-w02*sin(incl)*sin(u);
dBdip(2)=0;

% B=B0*.5*(1+(1+3*sin(incl)^2)^.5)*Bdip';
B=B0*Bdip';
dB=B0*dBdip';