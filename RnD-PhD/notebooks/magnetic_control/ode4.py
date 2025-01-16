function [T,Y] = ode4(ode, tspan, y0, options )
%ODE4   функция ODE4 реализует метод Рунге-Кутты 4 порядка
h = options.h; 

t0 = tspan(1);
tf = tspan(2);

T = t0:h:tf;
%T = [T.';tf];
T = T.';
nrowsT = size(T,1);
Y = zeros(nrowsT,length(y0));
Y(1,:) = y0(:).';
for norowT = 2:nrowsT   %количество строк в Т
    t = T(norowT - 1);
    y = Y(norowT-1,:).';
    h = T(norowT) - T(norowT - 1);
    
    k1 = ode(t,y);
    k2 = ode(t + (1/2)*h, y + (1/2)*h*k1);
    k3 = ode(t + (1/2)*h, y + (1/2)*h*k2);
    k4 = ode(t + h, y + h*k3);
    yn = y + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4);
   
    Y(norowT,:) = yn.';
end 
    
end