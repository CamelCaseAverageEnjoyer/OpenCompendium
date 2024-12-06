function [qr,w] = RK4FK(w,qr,u,t)
global w02 h
    [k11, k12]=rightSideFK(w, qr, u, t);
    [k21, k22]=rightSideFK(w+k11*h/2, qr+k12*h/2, u+w02*h/2, t+h/2);
    [k31, k32]=rightSideFK(w+k21*h/2, qr+k22*h/2, u+w02*h/2, t+h/2);
    [k41, k42]=rightSideFK(w+k31*h, qr+k32*h, u+w02*h, t+h);

    w=w+h/6*(k11+2*k21+2*k31+k41);

    qsatN=qr+h/6*(k12+2*k22+2*k32+k42);
    qr=qsatN/norm(qsatN,2);
end