function [out_alpha, out_AAdiag, out_w, out_qr,Control] = attitude_calculation(W0, QR0, alpha, t1, t2)
%Модель СОС "ЭМУ+магнитометр"

%очистка командной строки и удаление переменных
% clear all
% close all
% clc

%загрузка параметров
definitions

%загрузка параметров управления
control_parameters

%определение функции вычисления правой части
%сейчас ПД-регулятор. Возможные значения: @controlSM - скользящее
%управление, @controдDamp - демпфирование углового движения
control=@controlPD;

%управление: дипольный момент для демпфирования и ПД-регулятора,
%механический для скользящего управления
global mm
%орбитальная скорость
global w02
%шаг интегрирования
global h
h=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%начальные данные - конструируем кватернион и матрицу из углов и задаем скорость
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% qr=angle2quat(60.0*DEG_TO_RAD,130.0*DEG_TO_RAD,100.0*DEG_TO_RAD,'YZX')';
% qr=angle2quat(alpha0(2),alpha0(3),alpha0(1),'YZX')';
qr = QR0;
A=quat2dcm(qr');
%начальная скорость в радианах
% w=[10;10;10]*w02;
w=W0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%число шагов интегрирования
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time=3*3600/h;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%служебные массивы
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%запись результатов
ww=zeros(3,time+1);
AAdiag=zeros(3,time+1);
% AAdiag_kal=zeros(3,time+1);
tt=zeros(1,time);
QR = zeros(4,time+1);
%ФК
B_measurement=zeros(3,time); %Измерения магнитного поля
% Est=zeros(7,time);           %Оценка вектора состояния
Est=zeros(10,time);           %Оценка вектора состояния
% State=zeros(7,time);         %Реальное значение вектора состояния
State=zeros(10,time);         %Реальное значение вектора состояния
dw=0.1*pi/180*ones(3,1); %смещение ноля ДУСа
%отсчет итераций опредления ориентации
nom=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%размеры циклов измерения и управления
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
steps_est=1; %Число шагов для определения углового движения
steps_ctrl=5;%Число шагов для управления
steps_both=steps_est+steps_ctrl;%Суммарное количество

%запись начальных данных в служебные массивы
ww(:,1)=w;
AAdiag(1,1)=A(1,1);
AAdiag(2,1)=A(2,2);
AAdiag(3,1)=A(3,3);
% alpha_new = zeros(t2-t1+1);
alpha_new(1) = acos(A(1,1));
tt(1)=0;
mm=zeros(3,1);
%инициализация параметров Фильтра Калмана
Kalman_init; 
%основной цикл для моделирования работы СОС
K = [cos(alpha) sin(alpha) 0; -sin(alpha) cos(alpha) 0; 0 0 1]; %Орбит -> Опорн
% K = [-sin(alpha) cos(alpha) 0; 0 0 1; cos(alpha) sin(alpha) 0]; %Орбит -> Опорн
for i=1:t2-t1+1  %i=t1:t2
% i
%     A(:,i)=alpha0';
    %текущее время и аргумент широты
%     t=h*(i-1)+tt(1);
    t=h*(t1+i-1)+tt(1);
    u =  w02*t + u0;
    
    %вектор магнитного поля в ОСК
    Bmag=Bmag_orb(t,u);
    %вектор магнитного поля в ССК
    Bx = A*Bmag;

    %измерение или управление, в зависимости от текущего шага. Управление
    %строится на основе оценок фильтра Калмана
    if (rem(i,steps_both)<=steps_est)&&(rem(i,steps_both)>0)
        mm_last=mm;
        mm=zeros(3,1);
    else
%         A1 = A_est*K';
%         mm = control(w_est,Bx_est,A1);
        A1 = A*K';
        mm = control(w,Bx,A1);
    end
    
    if max(abs(mm))>0.3
        mm=0.3*mm/max(abs(mm));
    end

    %идеальное управление по реальному угловому движению
%     mm = control(w,Bx,A);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Интегрирование реального движения методом Рунге-Кутты 4-го порядка
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [qr,w] = RK4(w,qr,u,t);
    A=quat2dcm(qr'); 
    
    %Фильтр Калмана 
    if (rem(i,steps_both)<=steps_est)&&(rem(i,steps_both)>0)&&(i>1)
        %зашумления магнитного поля в ССК - моделирование измерений
        %шумовая составляющая магнитометра 100нТл+неточность модели МПЗ
        %200нТл и 100нТл смещение ноля измерений
%         B_measurement(:,nom)=normrnd(Bx+b_shift*1e-9*ones(3,1),100*1e-9*ones(3,1));
        B_measurement(:,nom)=normrnd(Bx+b_shift*1e-9*[1;0;0],100*1e-9*ones(3,1));
        b_sigma=100/norm (Bx)*1e-9;
%         Kalman_parameters{2,1}=diag([ones(3,1)*b_sigma^2;ones(3,1)*w_sigma^2]);
        Kalman_parameters{2,1}=diag([ones(3,1)*b_sigma^2]);
        W_measurement(:,nom)=normrnd(w+dw,w_sigma*ones(3,1));
        %получение интервала времени последних действий - время управления
        %или шаг, если это была итерация измерений
        if (rem(i,steps_both)==1)
            last_interval=steps_ctrl;
        else
            last_interval=1;
        end
        %Сохранение переменных в служебный массив
        State(:,nom+1)=[qr;w;dw];
        %восстановление управляющего момента, чтобы провести моделирование
        %в ФК
        mm=mm_last;  
        %Фильтр Калмана 
%         [Est(:,nom+1), P] = Kalman_mag(Est(:,nom), P, last_interval, B_measurement(:,nom),Bmag,Kalman_parameters,u,t);
%         [Est(:,nom+1), P] = Kalman_mag_dus(Est(:,nom), P, last_interval, [B_measurement(:,nom);W_measurement(:,nom)],Bmag,Kalman_parameters,u,t);
%         [Est(:,nom+1), P] = Kalman_mag_dus_dw(Est(:,nom), P, last_interval, [B_measurement(:,nom);W_measurement(:,nom)],Bmag,Kalman_parameters,u,t);
        mm=zeros(3,1);
        %получение оценки матрицы поворота, скорости, поля для управления
        A_est=quat2dcm(Est(1:4,nom+1)');
        w_est=Est(5:7,nom+1);
        %Оценка вектора мангитного поля в ОСК по оценкам фильтра Калмана
        Bmag=Bmag_orb(t+h,u+w02*h);
        Bx_est=quatrotate(Est(1:4,nom+1)',Bmag')';
        nom=nom+1;
    end
    
    %Сохранение переменных для построения графиков
    ww(:,i+1)=w;
    AAdiag(1,i+1)=A(1,1);
    AAdiag(2,i+1)=A(2,2);
    AAdiag(3,i+1)=A(3,3);
    
    alpha_new(i+1) = acos(A(1,1));
    QR(:,i+1)=qr;
    Control(:,i)=mm;
    
    tt(:,i+1)=t;
end

out_alpha = alpha_new(1:t2-t1+2);
out_AAdiag = AAdiag(:,1:t2-t1+2);

out_qr = QR(:,1:t2-t1+2);
out_w = ww(:,1:t2-t1+2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Color',[1 1 1])
% subplot(2,1,1);
% plot(tt/3600,ww(1,:)*180/pi,'-k',tt/3600,ww(2,:)*180/pi,':k',tt/3600,ww(3,:)*180/pi,'-.k');
% legend('\omega_1','\omega_2','\omega_3','Location','Best')
% xlabel('time, hours');
% ylabel('\Omega, deg/s');
% 
% subplot(2,1,2);
% plot(tt/3600,acos(AAdiag(1,:))*180/pi,'-k',tt/3600,acos(AAdiag(2,:))*180/pi,':k',tt/3600,acos(AAdiag(3,:))*180/pi,'-.k');
% legend('\gamma_1_1','\gamma_2_2','\gamma_3_3','Location','Best')
% xlabel('time, hours');
% ylabel('direction cosines, deg');
% ylim([0 180])
% % title('Параметры движения КА')
% 
% figure('Color',[1 1 1])
% tt1=1:nom;
% tt1=tt1*steps_both;
% %число значений, по которым производится вычисление среднеквадратической
% %оценки
% N=800;
% %вычисление ошибок оценки вектора угловой скорости
% Err_Omega1=(State(5,1:nom)'-Est(5,1:nom)')*180/pi;
% Err_Omega2=(State(6,1:nom)'-Est(6,1:nom)')*180/pi;
% Err_Omega3=(State(7,1:nom)'-Est(7,1:nom)')*180/pi;
% subplot(2,1,1);
% plot(tt1/3600,Err_Omega1,'-k',tt1/3600,Err_Omega2,':k',tt1/3600,Err_Omega3,'-.k');
% hold on
% %вычисление среднеквадратических отклонений
% sigma1=sqrt(sum(Err_Omega1(end-N:end).^2)/N);
% sigma2=sqrt(sum(Err_Omega2(end-N:end).^2)/N);
% sigma3=sqrt(sum(Err_Omega3(end-N:end).^2)/N);
% s='\Delta\omega_1; \sigma';
% leg1=sprintf('%s=%6.2e deg/s',s,sigma1);
% s='\Delta\omega_2; \sigma';
% leg2=sprintf('%s=%6.2e deg/s',s,sigma2);
% s='\Delta\omega_3; \sigma';
% leg3=sprintf('%s=%6.2e deg/s',s,sigma3);
% legend(leg1,leg2,leg3)
% xlabel('time, hours');
% ylabel('\Delta\Omega, deg/s');
% ylim([-1e-2 1e-2])
% % title('Ошибки определения угловой скорости')
% 
% % figure('Color',[1 1 1])
% %Вычисление углов ориентации из кватерниона
% [psi phi theta]=quat2angle(Est(1:4,1:nom)','YXZ');
% [psi_true phi_true theta_true]=quat2angle(State(1:4,1:nom)','YXZ');
% %вычисление разницы между реальными углами и их оценками
% delt1=(psi-psi_true)*180/pi;
% delt2=(phi-phi_true)*180/pi;
% delt3=(theta-theta_true)*180/pi;
% subplot(2,1,2);
% plot(tt1/3600,delt1,'-k',tt1/3600,delt2,':k',tt1/3600,delt3,'-.k');
% %вычисление среднеквадратических отклонений
% sigma1=sqrt(sum(delt1(end-N:end).^2)/N);
% sigma2=sqrt(sum(delt2(end-N:end).^2)/N);
% sigma3=sqrt(sum(delt3(end-N:end).^2)/N);
% s='\Delta\alpha; \sigma';
% leg1=sprintf('%s=%6.2e deg',s,sigma1);
% s='\Delta\beta; \sigma';
% leg2=sprintf('%s=%6.2e deg',s,sigma2);
% s='\Delta\gamma; \sigma';
% leg3=sprintf('%s=%6.2e deg',s,sigma3);
% legend(leg1,leg2,leg3)
% xlabel('time, hours');
% ylabel('Euler angles estimation error, deg');
% ylim([-5 5])
end
% % title('Ошибки определения ориентации')

% figure('Color',[1 1 1])
% plot(tt1/3600,(Est(8,1:nom)-dw(1)*ones(1,nom))*180/pi,'-k',tt1/3600,(Est(9,1:nom)-dw(2)*ones(1,nom))*180/pi,':k',tt1/3600,(Est(10,1:nom)-dw(1)*ones(1,nom))*180/pi,'-.k');
% ylim([-0.1 0.1])
% xlabel('Время, часы');
% ylabel('Ошибка  \Delta\omega, ^\circ/c');