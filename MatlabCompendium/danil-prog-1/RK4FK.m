function [x] = RK4FK(x, w, dt, u)
% integration using Runge-Kutta method

    dx1 = rightSideFK(x, w, u);
    dx2 = rightSideFK(x+dx1*dt/2, w, u);
    dx3 = rightSideFK(x+dx2*dt/2, w, u);
    dx4 = rightSideFK(x+dx3*dt, w, u);
    
    x = x+(dx1+2*dx2+2*dx3+dx4)*dt/6;
    
end