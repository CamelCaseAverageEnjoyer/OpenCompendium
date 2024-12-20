clc
close all
clear all
% 
% writerObj = VideoWriter('Animation.avi'); % File for saving the animation into avi file
% open(writerObj);

dt = 10; %Simulation time step
T_end = 30*24*60*60; % Simulation time
T = 0 : dt : T_end;


%% Загрузка всех параметров атмосферы ГОСТА

Ai=load('AtmGost_Ai_koef.txt');            %Таблицы коэффициентов из ГОСТа
Koef_150=load('AtmGost_Koef_120_500.txt');
Koef_500=load('AtmGost_Koef_500_1500.txt'); 
Activity_Sun=load('AtmGost_Activity.txt'); %Загрузка коэффициентов Солнечной активности из файла (для будущих времен нужно как-то аппроксимировать
Time_activity=juliandate([Activity_Sun(:,1) Activity_Sun(:,2) Activity_Sun(:,3) zeros(length(Activity_Sun(:,1)),3)]);
F107=Activity_Sun(:,4); 
F81=Activity_Sun(:,5);
Kp=Activity_Sun(:,6);

year_meas=2012; % Год, месяц и день получения телеметрии. Далее требудется для моделей магнитного поля и направления на Солнце
month_meas=3;
day_meas=12; 

t_begin=[year_meas month_meas day_meas 12 0 0]; % Время первого кадра телеметрии
t=0;
t_beging_year=[2012 1 1 18 0 0];

jd = juliandate(t_begin);

const = 2; %аэродинамичсекая константа
V = 7.9*1e3; %скорость набегающего потока
S_min = pi*0.187^2/4; %минимальная площадь грани спутника
S_max = 0.187*0.264; %максимальная площадь грани спутника
ro_const = 1e-12; %плотность атмосферы для расчета управления
m = 4.8; %масса спутника
u1_max=(1/2)*const*(S_max-S_min)*ro_const*(V^2)/m; %максимальная величина управления
% C4_control_value = 300; %красное значение для С4 
S_eci=[0.9857;-0.1596;-0.0542]; %Направление на Солнце в ИСК
%% Задание параметров аэродинамической силы
koeff=S_min*ro_const*V^2/m; % это произведение 
koeff_without_ro=koeff/ro_const; % это произведение S*V^2/m
% koeff=koeff_without_ro*ro_const;

%%
X_1 = zeros(6,length(T)); % Satellite position in Earth-centered inertial reference frame
X_2 = zeros(6,length(T)); % Debris position in Earth-centered inertial reference frame
X_3 = zeros(6,length(T)); % Debris position in Earth-centered inertial reference frame
xsat_diff_12 = zeros(6,length(T)); % Relative state vector
xsat_diff_13 = zeros(6,length(T)); % Relative state vector
mu = 3.986*10^14; % Gravitational parameter
R_earth = 6.371e6;


Trangle_length = 1000000; 

%% Chaser satellite parameters
Height = 500e3;           % Height of the satellite orbit
Rad = R_earth + Height;
incl_1 = 51.7*pi/180;     % Inclination
epsilon = 0;              % Eccentricity
phi = 0;                  % longitude of the ascending node
omega = 0;                % Argument of the pericenter
t_pi = 0;                 % pericenter time
t_cur = 0;                % current time
approx = 0;               % first step for Newton Method

u = Trangle_length/Rad/2;

 
[r, v] = orbitalMotionKeplerian_phi(mu, Rad, phi, incl_1, u);


X_1(1:3,1) = r;  
X_1(4:6,1) = v; % Absolute initial state vector of the satellite (ECI)

Omega = cross(X_1(1:3,1),X_1(4:6,1))/norm(X_1(1:3,1))^2; % Orbital angular velocity
A = orbital_dcm(X_1(:,1)); % Transition matrix to orbital reference frame

delta_i = (acos(0.02*(Rad/R_earth)^(7/2)/10+cos(51.7*pi/180))-51.7*pi/180)*180/pi;

%% 2nd satellite
Height = 500e3;           % Height of the satellite orbit
Rad = R_earth + Height;
incl_1 = 51.7*pi/180;     % Inclination
epsilon = 0;              % Eccentricity
phi = 0;                  % longitude of the ascending node
omega = 0;                % Argument of the pericenter
t_pi = 0;                 % pericenter time
t_cur = 0;                % current time
approx = 0;               % first step for Newton Method

u = - Trangle_length/Rad/2;

[r, v] = orbitalMotionKeplerian_phi(mu, Rad, phi, incl_1, u);


X_2(1:3,1) = r;  
X_2(4:6,1) = v; % Absolute initial state vector of the satellite (ECI)
i=1;
xsat_diff_12(:,i) = [A*(X_2(1:3,i) - X_1(1:3,i));A*((X_2(4:6,i) - X_1(4:6,i))-cross(Omega,(X_2(1:3,i) - X_1(1:3,i))))];

%% 3rd satellite
Height = 500e3;           % Height of the satellite orbit
Rad = R_earth + Height;
a = Trangle_length/Rad;
incl_2 = 51.7*pi/180-0.24*pi/180;%+delta_i*pi/180;     % Inclination
epsilon = 0;              % Eccentricity
phi = sqrt(3)/2*a*cos(incl_2-incl_1)/sin(incl_1);
u = -sqrt(3)/2*a*cos(incl_1)/sin(incl_2);                  % longitude of the ascending node
omega = 0;                % Argument of the pericenter
t_pi = 0;                 % pericenter time
t_cur = 0;                % current time
approx = 0;               % first step for Newton Method

 

[r, v] = orbitalMotionKeplerian_phi(mu, Rad, phi, incl_2, u);


X_3(1:3,1) = r;  
X_3(4:6,1) = v; % Absolute initial state vector of the satellite (ECI)

i=1;

xsat_diff_13(:,i) = [A*(X_3(1:3,i) - X_1(1:3,i));A*((X_3(4:6,i) - X_1(4:6,i))-cross(Omega,(X_3(1:3,i) - X_1(1:3,i))))];


figure('Color', [1 1 1])
plot3(xsat_diff_12(1,1),xsat_diff_12(2,1),xsat_diff_12(3,1),'ob','LineWidth',3)
hold on
plot3(0,0,0,'or','LineWidth',3)
plot3(xsat_diff_13(1,1),xsat_diff_13(2,1),xsat_diff_13(3,1),'og','LineWidth',3)
axis equal
grid on
xlabel('X, m')
ylabel('Y, m')
zlabel('Z, m')

if jd>floor(jd)+0.5 
    jd_activity=floor(jd)+0.5;
    UTC=(jd-(floor(jd)+0.5))/1.157401129603386e-05;
else
    jd_activity=floor(jd)-0.5;
    UTC=(jd-(floor(jd)-0.5))/1.157401129603386e-05;
end

i_activity=find(Time_activity==jd_activity);
ro(1)=atmosphere(X_1(1:3,1)/1000,S_eci,t_beging_year,jd,UTC,Ai,Koef_150,Koef_500,F81(i_activity),F107(i_activity),Kp(i_activity)); 

 
Control = zeros(3,1);
Time_length_plot = 0.9*90*60; % time for the length of the plot in animation
scrsz = get(0,'ScreenSize');
fig = figure('Color', [1 1 1],'Position',[1 1 scrsz(3) scrsz(4)])
load('topo.mat','topo','topomap2');
whos topo topomap2 

triangle_gap = 1000;
length_triangle = Trangle_length;
control_flag = 0;
%% Main cycle
for i = 2:1:length(T) 
    
    if i > 90*60/dt 
       hight_triangle = max(abs(xsat_diff_13(2,i-90*60/dt:i)));
       length_triangle = 2/sqrt(3)*hight_triangle;
    end
    
    C_constants = coord2const(xsat_diff_12(:,i-1), norm(Omega)); % Relative motion constants calculation 
    
            %% Атмосфера по ГОСТу
    jd=jd+dt*1.157401129603386e-05;
      
    if jd>floor(jd)+0.5 
        jd_activity=floor(jd)+0.5;
        UTC=(jd-(floor(jd)+0.5))/1.157401129603386e-05;
    else
        jd_activity=floor(jd)-0.5;
        UTC=(jd-(floor(jd)-0.5))/1.157401129603386e-05;
    end
    
    i_activity=find(Time_activity==jd_activity);
    
    options = odeset('RelTol',1e-7);
     
    %% 1-й аппарат
    ro(i)=atmosphere(X_1(1:3,i-1)/1000,S_eci,t_beging_year,jd,UTC,Ai,Koef_150,Koef_500,F81(i_activity),F107(i_activity),Kp(i_activity)); 
    A = orbital_dcm(X_1(:,i-1));
    if (abs(xsat_diff_12(1,i-1))<length_triangle) && control_flag ==1
        force(:,i) = [-S_max;0;0]*koeff_without_ro*ro(i)/S_min;
    else
        force(:,i) = [-S_min;0;0]*koeff_without_ro*ro(i)/S_min;
    end
     
%     force(:,i) = zeros(1,3);
    [~,X_new] = ode45(@(t,X) Right_part(t,X,A'*force(:,i)),[0:1:dt],X_1(:,i-1),options); % Integration of the satellite motion equations
    X_1(:,i) = X_new(end,:)';
    
    %% 2-й аппарат
    ro(i)=atmosphere(X_2(1:3,i-1)/1000,S_eci,t_beging_year,jd,UTC,Ai,Koef_150,Koef_500,F81(i_activity),F107(i_activity),Kp(i_activity)); 
    A = orbital_dcm(X_2(:,i-1));
%     abs(xsat_diff_12(1,i-1))-(length_triangle+triangle_gap)
    if abs(xsat_diff_12(1,i-1))>(length_triangle+triangle_gap)
        force(:,i) = [-S_max;0;0]*koeff_without_ro*ro(i)/S_min;
        control_flag = 1;
    else
        force(:,i) = [-S_min;0;0]*koeff_without_ro*ro(i)/S_min;
    end
%     force(:,i) = zeros(1,3);
    [~,X_new] = ode45(@(t,X) Right_part(t,X,A'*force(:,i)),[0:1:dt],X_2(:,i-1),options); % Integration of the debris motion equations
    X_2(:,i) = X_new(end,:)';
    
    %% 3-й аппарат
    ro(i)=atmosphere(X_3(1:3,i-1)/1000,S_eci,t_beging_year,jd,UTC,Ai,Koef_150,Koef_500,F81(i_activity),F107(i_activity),Kp(i_activity));
    A = orbital_dcm(X_3(:,i-1));
    if abs(xsat_diff_12(1,i-1))>(length_triangle+triangle_gap)
        force(:,i) = [-S_max;0;0]*koeff_without_ro*ro(i)/S_min;
        control_flag = 1;
    else
        force(:,i) = [-S_min;0;0]*koeff_without_ro*ro(i)/S_min;
    end
%     force(:,i) = zeros(1,3);
    [~,X_new] = ode45(@(t,X) Right_part(t,X,A'*force(:,i)),[0:1:dt],X_3(:,i-1),options); % Integration of the debris motion equations
    X_3(:,i) = X_new(end,:)';
    
    %% Вычисление расстояний
    Omega = cross(X_1(1:3,i),X_1(4:6,i))/norm(X_1(1:3,i))^2; % Orbital angular velocity
    A = orbital_dcm(X_1(:,i)); % Transition matrix to orbital reference frame
    xsat_diff_12(:,i) = [A*(X_2(1:3,i) - X_1(1:3,i));A*((X_2(4:6,i) - X_1(4:6,i))-cross(Omega,(X_2(1:3,i) - X_1(1:3,i))))]; % Relative state vector in orbital reference frame
    xsat_diff_13(:,i) = [A*(X_3(1:3,i) - X_1(1:3,i));A*((X_3(4:6,i) - X_1(4:6,i))-cross(Omega,(X_3(1:3,i) - X_1(1:3,i))))]; % Relative state vector in orbital reference frame

    
    
    
    
    
    i/length(T)
    %% Draw animation
%     if rem(i,int16(length(T)/500))==0 %% Annimation is plotted each length(T)/500 steps
%         i/length(T)
%         
%         if T(i)<Time_length_plot
%             interval = [1:i];
%         else
%             i_start = int16(Time_length_plot/dt);
%             interval = [i-i_start:i];
%         end
%         
%         plot3(xsat_diff_12(1,interval),xsat_diff_12(2,interval),xsat_diff_12(3,interval),'b','LineWidth',4)
%         hold on 
%         
%         plot3(xsat_diff_12(1,interval),xsat_diff_12(2,interval),xsat_diff_12(3,interval),'ob','LineWidth',4)
%         plot3(xsat_diff_13(1,interval),xsat_diff_13(2,interval),xsat_diff_13(3,interval),'og','LineWidth',4)
%         
%         plot3([0 xsat_diff_12(1,i) xsat_diff_13(1,i) 0],[0 xsat_diff_12(2,i) xsat_diff_13(2,i) 0],[0 xsat_diff_12(3,i) xsat_diff_13(3,i) 0],'k','LineWidth',2)
%         plot3(0,0,0,'or','LineWidth',3)
%         axis equal
%         grid on
%         legend('1st satellite','2nd satellite','3rd satellite','Location','NorthEast')
% %
%         hold off 
% %         xlim([-1.1e6 0])
% %         ylim([-8.5e5 8.5e5]) 
% %         zlim([-100 100]) 
%         xlabel('x, m','FontSize',18)
%         ylabel('y, m','FontSize',18)
%         zlabel('z, m','FontSize',18)
%         title('Relative trajectory','FontSize',20)
%         set(gca,'FontSize',18)
%         hold off
%         pause(0.1) 
        
%         subplot(1,2,2)
%         plot3(X_1(1,interval),X_1(2,interval),X_1(3,interval),'b','LineWidth',2)
%         hold on
%         plot3(X_2(1,interval),X_2(2,interval),X_2(3,interval),'r','LineWidth',2)
%         plot3(X_3(1,interval),X_3(2,interval),X_3(3,interval),'g','LineWidth',2)
% 
%         plot3(X_1(1,i),X_1(2,i),X_1(3,i),'ob','LineWidth',3)
%         hold on
%         plot3(X_2(1,i),X_2(2,i),X_2(3,i),'or','LineWidth',3)
%         plot3(X_3(1,i),X_3(2,i),X_3(3,i),'og','LineWidth',3) 
%         plot3([X_1(1,i) X_2(1,i) X_3(1,i) X_1(1,i)],[X_1(2,i) X_2(2,i) X_3(2,i) X_1(2,i)],[X_1(3,i) X_2(3,i) X_3(3,i) X_1(3,i)],'k','LineWidth',2)
%  
%         [x,y,z] = sphere(50);
%         [x, y, z] = EarthRotation(x, y, z, i*dt);
%         props.AmbientStrength = 0.1;
%         props.DiffuseStrength = 1;
%         props.SpecularColorReflectance = .5;
%         props.SpecularExponent = 20;
%         props.SpecularStrength = 1;
%         props.FaceColor= 'texture';
%         props.EdgeColor = 'none';
%         props.FaceLighting = 'phong';
%         props.Cdata = topo;
%         surface(x*R_earth,y*R_earth,z*R_earth,props);
%         light('position',[1 -1 1]);
%         light('position',[1.5 -0.5 0.5], 'color', [.6 .2 .2]);
%         axis equal
%         view(45,45)
%         xlabel('X, m','FontSize',18)
%         ylabel('Y, m','FontSize',18)
%         zlabel('Z, m','FontSize',18)
%         legend('1st satellite','2nd satellite','3rd satellite','Location','NorthEast')
%         grid on
%         xlim([-7e6 7e6])
%         ylim([-7e6 7e6]) 
%         zlim([-7e6 7e6]) 
%         set(gca,'FontSize',18)
%         hold off
%         
%         pause(0.1)  
%         F = getframe(fig);  % for saving the animation into the avifile
%         writeVideo(writerObj,F);
%     end

end
% close(writerObj);

figure('Color', [1 1 1])
plot3(X_1(1,:),X_1(2,:),X_1(3,:),'b','LineWidth',2)
hold on
plot3(X_2(1,:),X_2(2,:),X_2(3,:),'r','LineWidth',2)
plot3(X_3(1,:),X_3(2,:),X_3(3,:),'g','LineWidth',2)

plot3(X_1(1,end),X_1(2,end),X_1(3,end),'ob','LineWidth',3)
hold on
plot3(X_2(1,end),X_2(2,end),X_2(3,end),'or','LineWidth',3)
plot3(X_3(1,end),X_3(2,end),X_3(3,end),'og','LineWidth',3)
plot3([X_1(1,end) X_2(1,end) X_3(1,end) X_1(1,end)],[X_1(2,end) X_2(2,end) X_3(2,end) X_1(2,end)],[X_1(3,end) X_2(3,end) X_3(3,end) X_1(3,end)],'k','LineWidth',2)
load('topo.mat','topo','topomap1');
whos topo topomap1
[x,y,z] = sphere(50);
props.AmbientStrength = 0.1;
props.DiffuseStrength = 1;
props.SpecularColorReflectance = .5;
props.SpecularExponent = 20;
props.SpecularStrength = 1;
props.FaceColor= 'texture';
props.EdgeColor = 'none';
props.FaceLighting = 'phong';
props.Cdata = topo;
surface(x*R_earth,y*R_earth,z*R_earth,props);

% Add lights.
light('position',[-1 0 1]);
light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);
axis equal
view(3)
xlabel('X, m')
ylabel('Y, m')
zlabel('Z, m')
title('Motion in ECI')
legend('Chaser','Debris')
grid on

figure('Color', [1 1 1])
plot3(xsat_diff_12(1,end),xsat_diff_12(2,end),xsat_diff_12(3,end),'ob','LineWidth',3)
hold on
plot3(0,0,0,'or','LineWidth',3)
plot3(xsat_diff_12(1,:),xsat_diff_12(2,:),xsat_diff_12(3,:),'b','LineWidth',2)
plot3(xsat_diff_13(1,:),xsat_diff_13(2,:),xsat_diff_13(3,:),'g','LineWidth',2)
plot3(xsat_diff_13(1,end),xsat_diff_13(2,end),xsat_diff_13(3,end),'ok','LineWidth',3)
axis equal
grid on
xlabel('X, m')
ylabel('Y, m')
zlabel('Z, m')

figure('Color', [1 1 1])
plot(T/86400, xsat_diff_13(2,:)*2/sqrt(3),'g','LineWidth',2)
grid on
xlabel('Time, days')
ylabel('Distance_13, km')

figure('Color', [1 1 1])
plot(T/86400,xsat_diff_12(1,:)/1000,'b','LineWidth',2)
grid on
xlabel('Time, days')
ylabel('Distance_12, km')

for i=1:1:length(X_3(1,:))
    Hight_1(i) = norm(X_1(1:3,i))-R_earth;
end

figure('Color', [1 1 1])
plot(T/86400,Hight_1/1000,'b','LineWidth',2)
grid on
xlabel('Time, days')
ylabel('Hight, km')

figure('Color', [1 1 1])
plot(T/86400,ro,'b','LineWidth',2)
grid on
xlabel('Time, days')
ylabel('Density, kg/m^3')
