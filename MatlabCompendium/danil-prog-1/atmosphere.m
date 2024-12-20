function [ro] = atmosphere(xsat_eci,S_eci,t_beging_year,jd,UTC,Ai,Koef_150,Koef_500,F81,F107,Kp)
%ATMOSPHERE Summary of this function goes here
%   Detailed explanation goes here
% xsat_eci - вектор до точки в инерциальной системе координат (ИСК)
% S_eci - вектор направления на Солнце в ИСК
% t_beging_year - время начала текущего года
% jd - текущий Юлианский день
% UTC - всеобщее скоординированное время
% Ai, Koef_150,Koef_500,F81,F107,Kp - коэффициенты из ГОСТа (описание лучше
% смотреть в ГОСТе)

%% Задание коэффициентов
R_Earth=6.4e3; %Радиус Земли
r=norm(xsat_eci); %модуль радиус вектора
h=(r-R_Earth);    % Высота над сферической Землей
ro0=1.58868e-8;   % плотность ночной атмосферы на высоте 120 км, кг/м ;
w_Earth=7.292115e-5; %угловая скорость вращения Земли, рад/с;
delta_Sun=atan(S_eci(3)/S_eci(2)); % один из углов направления на Солнце

date=floor(jd)-floor(juliandate(t_beging_year));
if S_eci(2)>=0
    alpha_Sun=acos(S_eci(1));
else
    alpha_Sun=2*pi-acos(S_eci(1));
end

[a,b,c,n,fi1,d,e,l,F0]=Koefficients(Koef_150,Koef_500,F81,h); %Функция, которая выбирает коэффициенты для заданной высоты h

S_grin_midnight=grin_midnight(jd); % Вычисление звездного времени в гринвическую полночь
beta=alpha_Sun-S_grin_midnight-w_Earth*UTC+fi1; % Далее все вычисления по формулам ГОСТа
CosFI=(xsat_eci(3)*sin(delta_Sun)+(xsat_eci(1)*cos(beta)+xsat_eci(2)*sin(beta))*cos(delta_Sun))/norm(xsat_eci);
Ad=Ai(1)+Ai(2)*date+Ai(3)*date^2+Ai(4)*date^3+Ai(5)*date^4+Ai(6)*date^5+Ai(7)*date^6+Ai(8)*date^7+Ai(9)*date^8;

roN=ro0*exp(a(1)+a(2)*h+a(3)*h^2+a(4)*h^3+a(5)*h^4+a(6)*h^5+a(7)*h^6); 
K0=1+(l(1)+l(2)*h+l(3)*h^2+l(4)*h^3+l(5)*h^4)*(F81-F0)/F0;
K1=(c(1)+c(2)*h+c(3)*h^2+c(4)*h^3+c(5)*h^4)*(sqrt((CosFI+1)/2)^(n(1)+n(2)*h+n(3)*h^2));
K2=(d(1)+d(2)*h+d(3)*h^2+d(4)*h^3+d(5)*h^4)*Ad;
K3=(b(1)+b(2)*h+b(3)*h^2+b(4)*h^3+b(5)*h^4)*(F107-F81)/(F81+abs(F107-F81));
K4=(e(1)+e(2)*h+e(3)*h^2+e(4)*h^3+e(5)*h^4)*(e(6)+e(7)*Kp+e(8)*Kp^2+e(9)*Kp^3);

ro=roN*K0*(1+K1+K2+K3+K4);

end