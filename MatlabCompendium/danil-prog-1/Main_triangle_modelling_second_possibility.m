clc
close all
clear all

% writerObj = VideoWriter('Animation.avi'); % File for saving the animation into avi file
% open(writerObj);

dt = 10; %Simulation time step
T_end = 10*60*60; % Simulation time
T = 0 : dt : T_end; 


X_1 = zeros(6,length(T)); % Satellite position in Earth-centered inertial reference frame
X_2 = zeros(6,length(T)); % Debris position in Earth-centered inertial reference frame
X_3 = zeros(6,length(T)); % Debris position in Earth-centered inertial reference frame
xsat_diff_12 = zeros(6,length(T)); % Relative state vector
xsat_diff_13 = zeros(6,length(T)); % Relative state vector
mu = 3.986*10^14; % Gravitational parameter
R_earth = 6.371e6;

%% Chaser satellite parameters 
Hight = 500e3; % Hight of the satellite orbit
Rad = R_earth + Hight;
Trangle_length = 1000000;
Angle_triangle = asin(Trangle_length/Rad/2);
incl_1 = 51.7*pi/180; % Inclination
% X_1(:,1) = [Rad*cos(Angle_triangle); Rad*sin(Angle_triangle); 0; -sqrt(mu/Rad)*sin(Angle_triangle); sqrt(mu/Rad)*cos(Angle_triangle); 0]; % Absolute initial state vector of the satellite
X_1(:,1) = [Rad; 0; 0; 0; sqrt(mu/Rad); 0]; % Absolute initial state vector of the satellite

[r, v, D] = orbitalMotionKeplerian(mu, Rad, 0, 0, 0, incl_1, 0, 0, 0)


X_1(1:3,1) = r; % Inclined reference frame
X_1(4:6,1) = v;

Omega = cross(X_1(1:3,1),X_1(4:6,1))/norm(X_1(1:3,1))^2; % Orbital angular velocity
A = orbital_dcm(X_1(:,1)); % Transition matrix to orbital reference frame

%% Debris object parameters 
Hight = 500e3; % Hight of the debris orbit
Rad = R_earth + Hight;
% X_2(:,1) = [Rad; 0; 0; 0; sqrt(mu/Rad)*0.99915; 0]; % Absolute initial state vector of the satellite
% C_rel = [0; 0; 0; -1000000; 0; 0]; % Linear motion equation parameters
% dX = trajectory(norm(Omega),C_rel,0); % Relative vector
% X_2(:,1) = [X_1(1:3,1)+A'*dX(1:3)';X_1(4:6,1)+A'*dX(4:6)'+cross(Omega,A'*dX(1:3)')]; % Absolute initial state vector of the satellite
% X_2(:,1) = [Rad*cos(Angle_triangle); - Rad*sin(Angle_triangle); 0; sqrt(mu/Rad)*sin(Angle_triangle); sqrt(mu/Rad)*cos(Angle_triangle); 0]; % Absolute initial state vector of the satellite
[r, v, D] = orbitalMotionKeplerian(mu, Rad, 0, Angle_triangle, 0, incl_1, -150, 0, 0)
% incl_1 = 51.7*pi/180; % Inclination
% A_incl = [1 0 0;
%           0 cos(incl_1) sin(incl_1); 
%           0 -sin(incl_1) cos(incl_1)]; % Transition matrix

X_2(1:3,1) = r; % Inclined reference frame
X_2(4:6,1) = v;


i=1;

xsat_diff_12(:,i) = [A*(X_2(1:3,i) - X_1(1:3,i));A*((X_2(4:6,i) - X_1(4:6,i))-cross(Omega,(X_2(1:3,i) - X_1(1:3,i))))];

%% 3rd satellite
% C_rel = [0; 0; 0; -1000000; 0; 1000000*sqrt(3)/2]; % Linear motion equation parameters
% dX = trajectory(norm(Omega),C_rel,0); % Relative vector
% X_3(:,1) = [X_1(1:3,1)+A'*dX(1:3)';X_1(4:6,1)+A'*dX(4:6)'+cross(Omega,A'*dX(1:3)')]; % Absolute initial state vector of the satellite
% Angle_triangle = asin(Trangle_length*sqrt(3)/2/Rad);
% X_3(:,1) = [Rad*cos(Angle_triangle); 0; Rad*sin(Angle_triangle); 0; sqrt(mu/Rad); 0]; % Absolute initial state vector of the satellite
[r, v, D] = orbitalMotionKeplerian(mu, Rad, 0, 4*Angle_triangle, 0, incl_1, 150, 0, 0)
incl_1 = 51.7*pi/180; % Inclination
A_incl = [1 0 0;
          0 cos(incl_1) sin(incl_1);
          0 -sin(incl_1) cos(incl_1)]; % Transition matrix

X_3(1:3,1) = r; % Inclined reference frame
X_3(4:6,1) = v;

i=1;

xsat_diff_13(:,i) = [A*(X_2(1:3,i) - X_1(1:3,i));A*((X_2(4:6,i) - X_1(4:6,i))-cross(Omega,(X_2(1:3,i) - X_1(1:3,i))))];


Control = zeros(3,1);
Time_length_plot = 0.9*90*60; % time for the length of the plot in animation
scrsz = get(0,'ScreenSize');
fig = figure('Color', [1 1 1],'Position',[1 1 scrsz(3) scrsz(4)])
load('topo.mat','topo','topomap1');
whos topo topomap1 

%% Main cycle
for i = 2:1:length(T) 
    force(:,i) = zeros(3,1);
    C_constants = coord2const(xsat_diff_12(:,i-1), norm(Omega)); % Relative motion constants calculation 
    
    A = orbital_dcm(X_1(:,i-1));
    
    [~,X_new] = ode45(@(t,X) Right_part(t,X,A'*force(:,i)),[0:1:dt],X_1(:,i-1)); % Integration of the satellite motion equations
    X_1(:,i) = X_new(end,:)';
    
    force(:,i)=zeros(3,1);
    [~,X_new] = ode45(@(t,X) Right_part(t,X,force(:,i)),[0:1:dt],X_2(:,i-1)); % Integration of the debris motion equations
    X_2(:,i) = X_new(end,:)';
    
    force(:,i)=zeros(3,1);
    [~,X_new] = ode45(@(t,X) Right_part(t,X,force(:,i)),[0:1:dt],X_3(:,i-1)); % Integration of the debris motion equations
    X_3(:,i) = X_new(end,:)';
    
    Omega = cross(X_1(1:3,i),X_1(4:6,i))/norm(X_1(1:3,i))^2; % Orbital angular velocity
    A = orbital_dcm(X_1(:,i)); % Transition matrix to orbital reference frame
    xsat_diff_12(:,i) = [A*(X_2(1:3,i) - X_1(1:3,i));A*((X_2(4:6,i) - X_1(4:6,i))-cross(Omega,(X_2(1:3,i) - X_1(1:3,i))))]; % Relative state vector in orbital reference frame
    xsat_diff_13(:,i) = [A*(X_3(1:3,i) - X_1(1:3,i));A*((X_3(4:6,i) - X_1(4:6,i))-cross(Omega,(X_3(1:3,i) - X_1(1:3,i))))]; % Relative state vector in orbital reference frame


    %% Draw animation
    if rem(i,int16(length(T)/500))==0 %% Annimation is plotted each length(T)/500 steps
        i/length(T)
        
        if T(i)<Time_length_plot
            interval = [1:i];
        else
            i_start = int16(Time_length_plot/dt);
            interval = [i-i_start:i];
        end
        subplot(1,2,1)
%         plot3(xsat_diff_12(1,interval),xsat_diff_12(2,interval),xsat_diff_12(3,interval),'b','LineWidth',2)
        
        
        plot3(xsat_diff_12(1,i),xsat_diff_12(2,i),xsat_diff_12(3,i),'ob','LineWidth',3)
        hold on 
        plot3(xsat_diff_13(1,i),xsat_diff_13(2,i),xsat_diff_13(3,i),'og','LineWidth',3)
        
        plot3([0 xsat_diff_12(1,i) xsat_diff_13(1,i) 0],[0 xsat_diff_12(2,i) xsat_diff_13(2,i) 0],[0 xsat_diff_12(3,i) xsat_diff_13(3,i) 0],'k','LineWidth',2)
        plot3(0,0,0,'or','LineWidth',3)
        axis equal
        grid on
        hold off 
%         xlim([-700 50])
%         ylim([-50 50]) 
%         zlim([-100 100]) 
        xlabel('X, m')
        ylabel('Y, m')
        zlabel('Z, m')
        
        subplot(1,2,2)
        plot3(X_1(1,interval),X_1(2,interval),X_1(3,interval),'b','LineWidth',2)
        hold on
        plot3(X_2(1,interval),X_2(2,interval),X_2(3,interval),'r','LineWidth',2)
        plot3(X_3(1,interval),X_3(2,interval),X_3(3,interval),'g','LineWidth',2)

        plot3(X_1(1,i),X_1(2,i),X_1(3,i),'ob','LineWidth',3)
        hold on
        plot3(X_2(1,i),X_2(2,i),X_2(3,i),'or','LineWidth',3)
        plot3(X_3(1,i),X_3(2,i),X_3(3,i),'og','LineWidth',3) 
        plot3([X_1(1,i) X_2(1,i) X_3(1,i) X_1(1,i)],[X_1(2,i) X_2(2,i) X_3(2,i) X_1(2,i)],[X_1(3,i) X_2(3,i) X_3(3,i) X_1(3,i)],'k','LineWidth',2)

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
        light('position',[-1 0 1]);
        light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);
        axis equal
        view(2)
        xlabel('X, m')
        ylabel('Y, m')
        zlabel('Z, m')
%         legend('1st satellite','2nd satellite','3rd satellite')
        grid on
        hold off
        
        pause(0.1)   
%         F = getframe(fig);  % for saving the animation into the avifile
%         writeVideo(writerObj,F);
    end

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
axis equal
grid on
xlabel('X, m')
ylabel('Y, m')
zlabel('Z, m')