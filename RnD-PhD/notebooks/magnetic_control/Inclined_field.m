function Bmag = Inclined_field(t,r)
    global B0 t0scal k0
    global DEG_TO_RAD
    
    %получаем текущее время
    tcur=addtodate(t0scal,round(1000*t),'millisecond');
    datecur = datevec(tcur);
    %вычисляем юлианскую дату в днях
    JD=juliandate(datecur); %сверено с комплексом
    
    %перевод вектора направления диполя от гринвической системы к инерциальной текущей даты
	theta = 280.46061837 + 360.98564736629 * (JD - 2451545.0);
    theta = theta * DEG_TO_RAD;
    
	
	k(1)=k0(1)*cos(theta)-k0(2)*sin(theta);
	k(2)=k0(1)*sin(theta)+k0(2)*cos(theta);
	k(3)=k0(3);
    
    Bmag=B0*(k'-3*(k*r)*r);
end
