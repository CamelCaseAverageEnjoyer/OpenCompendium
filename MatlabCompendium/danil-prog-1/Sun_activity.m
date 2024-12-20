function [nom] = Sun_activity(F81)
%SUN_ACTIVITY Summary of this function goes here
%   Detailed explanation goes here
if F81>0&&F81<=87.5
    nom=1;
end
if F81>87.5&&F81<=112.5
    nom=2;
end
if F81>112.5&&F81<=137.5
    nom=3;
end
if F81>137.5&&F81<=162.5
    nom=4;
end
if F81>162.5&&F81<=187.5
    nom=5;
end
if F81>187.5&&F81<=225
    nom=6;
end
if F81>225
    nom=7;
end



end

