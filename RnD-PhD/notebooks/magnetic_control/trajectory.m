function [X]=trajectory(w,C,t) 
X(:,1)=-3*C(1)*w*t+2*C(2)*cos(w*t)-2*C(3)*sin(w*t)+C(4);
X(:,2)=C(5)*sin(w*t)+C(6)*cos(w*t);
X(:,3)=2*C(1)+C(2)*sin(w*t)+C(3)*cos(w*t);
X(:,4)=-3*C(1)*w-2*C(2)*w*sin(w*t)-2*C(3)*w*cos(w*t);
X(:,5)=C(5)*w*cos(w*t)-C(6)*w*sin(w*t);
X(:,6)=C(2)*w*cos(w*t)-C(3)*w*sin(w*t);
end
