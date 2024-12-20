function [a,b,c,n,fi,d,e,l,F0] = Koefficients(Koef_150,Koef_500,F81,h)
%KOEFFICIENTS Summary of this function goes here
%   Detailed explanation goes here
%% Определение ближайшего уровня солнечной активности


nom=Sun_activity(F81);
F=[75:25:250];
F0=F(nom);
for i=1:1:7
    if h<Koef_500(2,nom)
        a(i)=Koef_150(2+i,nom);
    else
        a(i)=Koef_500(2+i,nom);
    end
end
    
for i=1:1:5
    if h<Koef_500(10,nom)
        b(i)=Koef_150(10+i,nom);
    else
        b(i)=Koef_500(10+i,nom);
    end
end
    
for i=1:1:5
    if h<Koef_500(16,nom)
        c(i)=Koef_150(16+i,nom);
    else
        c(i)=Koef_500(16+i,nom);
    end
end

for i=1:1:3
    if h<Koef_500(16,nom)
        n(i)=Koef_150(21+i,nom);
    else
        n(i)=Koef_500(21+i,nom);
    end
end

    if h<Koef_500(16,nom)
        fi=Koef_150(25,nom);
    else
        fi=Koef_500(25,nom);
    end
    
for i=1:1:5
        d(i)=Koef_150(26+i,nom);
end
        
for i=1:1:13
    if h<Koef_500(32,nom)
        e(i)=Koef_150(32+i,nom);
    else
        e(i)=Koef_500(32+i,nom);
    end
end

for i=1:1:5
    if h<Koef_500(46,nom)
        l(i)=Koef_150(46+i,nom);
    else
        l(i)=Koef_500(46+i,nom);
    end
end


end

