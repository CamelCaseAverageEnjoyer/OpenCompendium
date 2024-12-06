clc
close all
clear all

N = 12; %Amount of satellites

dt = 10; %Simulation time step
T_end = 2*60*60; % Simulation time
T = 0 : dt : T_end; 


% X_1 = zeros(6,length(T)); % First Satellite position in Earth-centered inertial reference frame
% X_2 = zeros(6,length(T)); % Second Satellite position in Earth-centered inertial reference frame
% xsat_diff_12 = zeros(6,length(T)); % Relative state vector
mu = 3.986*10^14; % Gravitational parameter
R_earth = 6.371e6; %

%% Chaser satellite parameters 
Hight = 500e3; % Hight of the satellite orbit
Rad = R_earth + Hight;
incl_1 = 51.7*pi/180; % Inclination
epsilon = 0; %excentricity
phi = 0; % longitude of the ascending node
omega = 0; % argument of the pericenter
t_pi = 0; % pericenter time
t_cur = 0; % current time
approx = 0; % first step for Newton Method
[r, v, D] = orbitalMotionKeplerian(mu, Rad, epsilon, phi, omega, incl_1, t_pi, t_cur, approx);

m = 4*1e-3; % mass of ChipSat
M = 3; % mass of main sat
mu_0 = 4*pi*1e-7; % permeability of free space
Mu = [0.5;0.5;0.5]; % magnetic moment of main sat
J_3U = diag([0.005; 0.025; 0.025]);
J_ChipSat = diag([2.5*1e-7; 1.25*1e-7; 1.25*1e-7]);

X_1(1:3,1) = r; 
X_1(4:6,1) = v; % Absolute initial state vector of the satellite

Omega(:,1) = cross(X_1(1:3,1),X_1(4:6,1))/norm(X_1(1:3,1))^2; % Orbital angular velocity
A = orbital_dcm(X_1(:,1)); % Transition matrix to orbital reference frame

%initial angular state for CubeSat
ang1(1) = 5.0*pi/180;
ang2(1) = 8.0*pi/180;
ang3(1) = 3.0*pi/180;
qr(:,1)=angle2quat(ang1(1),ang2(1),ang3(1),'YZX')';
w_ang(:,1) = [0.01; 0.009; 0.011];
X_1(7:10,1) = qr(:,1);
X_1(11:13,1) = w_ang(:,1);

%% Other satellites' parameters

R0_launch = zeros(N,3);
V0_launch(1:N,1:3) = (rand(N,3)*1e-3)-(0.25*1e-3*ones(N,3)); %0.5*1e-3*ones(N,3);
V0_launch(1:N/4,1) = (rand(N/4,1)*1e-3) + (0.5*1e-2*ones(N/4,1)); %0.5*1e-2*ones(N/4,1); %(rand(N/4,1)+0.5)*1e-2; %velocity ox positive direction
V0_launch(N/4+1:N/2,3) = (rand(N/4,1)*1e-3) + (0.5*1e-2*ones(N/4,1)); %0.5*1e-3*ones(N/4,1); %(rand(N/4,1)+0.5)*1e-3; %velocity oz positive direction
V0_launch(N/2+1:3*N/4,1) = -(rand(N/4,1)*1e-3) - (0.5*1e-2*ones(N/4,1)); %;%-(rand(N/4,1)+0.5)*1e-2; %velocity ox negative direction
V0_launch(3*N/4+1:N,3) = -(rand(N/4,1)*1e-3) - (0.5*1e-2*ones(N/4,1));%-0.5*1e-3*ones(N/4,1);%-(rand(N/4,1)+0.5)*1e-3; %velocity oz negative direction

RV0_launch = [R0_launch, V0_launch];
dt_launch = 5; %time between launches
tend_launch = N/4*dt_launch; %time when control begins
R0 = zeros(N,3);
V0 = zeros(N,3);
force_disturb = cell(N,N);

for j=1:N/4
    for k=j:N/4:N
    y0 = RV0_launch(k,:);
    t0 = (j-1)*dt_launch;
    [~,out_rv] = ode45(@(t,X) Hill_equation(t,X,norm(Omega(:,1))),[t0:1:tend_launch],y0);
    dX0(k,:) = out_rv(end,:);  
    % Absolute initial state vector of satellites
    X_k(:,1,k) = [X_1(1:3,1)+A'*dX0(k,1:3)';X_1(4:6,1)+A'*dX0(k,4:6)'+cross(Omega(:,1),A'*dX0(k,1:3)')]; 
    % Relative state vector calculation (relative to main sat)
    xsat_diff(:,1,k) = [A*(X_k(1:3,1,k) - X_1(1:3,1));A*((X_k(4:6,1,k) - X_1(4:6,1))-cross(Omega(:,1),(X_k(1:3,1,k) - X_1(1:3,1))))];
    %trajectory without control
    [~,xsat] = ode45(@(t,X) Hill_equation(t,X,norm(Omega(:,1))),[0:1:T_end],xsat_diff(:,1,k)); 
    xsat_k(:,:,k) = xsat;
    end
end

%initial angular state for ChipSat
for j = 1:N
    w_ang_k(:,1,j) = w_ang(:,1);
    qr_k(:,1,j) = qr(:,1);
    ang1_k(1,j) = ang1(1);
    ang2_k(1,j) = ang2(1); 
    ang3_k(1,j) = ang3(1);
    X_k(7:10,1,j) = qr_k(:,1,j);
    X_k(11:13,1,j) = w_ang_k(:,1,j);
end
% for k=1:N
%     for j = 1:N
%         d_rel{k,j}=xsat_diff(1:3,1,k)-xsat_diff(1:3,1,j);
%     end
% end
figure
for k=1:N
    plot3(xsat_k(:,1,k),xsat_k(:,2,k),xsat_k(:,3,k),'LineWidth',1)
    colormap(hsv(2*k+5))
    hold on
end
grid on
% axis equal
xlabel('x, m') 
ylabel('y, m')
zlabel('z, m')

procent = 5/6;

%% Main cycle
for i = 2:1:length(T) 
    i
    Force(:,i-1) = zeros(3,1);
    B_sum(:,i-1) = zeros(3,1);
    for j = 1:N
        for k = 1:N
        C_constants(:,i-1,k) = coord2const(xsat_diff(:,i-1,k), norm(Omega(:,i-1))); % Relative motion constants calculation 
        end
        C_1(1:N) = sort(abs(C_constants(1,i-1,:)));
        if norm(C_1(1:procent*N)) > 0.05
            force_ideal(i-1,j) = -m*C_constants(1,i-1,j)*norm(Omega(:,i-1))/(2*dt); % desired force
            d = norm(xsat_diff(1:3,i-1,j));
            d_r = xsat_diff(1:3,i-1,j);
            
            [MU] = magnetic_dipole(force_ideal(i-1,j), d_r, Mu);
            mu(1:3,i-1,j) = MU;
            value_mu(i-1,j) = norm(mu(:,i-1,j));
            value_mu_2_desired(i-1,j) = norm(mu(:,i-1,j));
            mu_max_value = 0.005;
            if value_mu(i-1,j) > mu_max_value
               mu(:,i-1,j) = mu_max_value*(mu(:,i-1,j)/norm(mu(:,i-1,j)));
               value_mu(i-1,j) = norm(mu(:,i-1,j));
               force_real(:,i-1,j) = (3*mu_0/(4*pi*d^5))*(mu(:,i-1,j)'*Mu*d_r + Mu'*d_r*mu(:,i-1,j) + mu(:,i-1,j)'*d_r*Mu - (5/d^2)*(Mu'*d_r)*(mu(:,i-1,j)'*d_r)*d_r) ;
               force(:,i-1,j) = force_real(:,i-1,j);
               B(:,i-1,j) = (mu_0/(4*pi*d^5))*(3*mu(:,i-1,j)'*d_r*d_r - d^2*mu(:,i-1,j));
            else
               force(:,i-1,j) = (3*mu_0/(4*pi*d^5))*(mu(:,i-1,j)'*Mu*d_r + Mu'*d_r*mu(:,i-1,j) + mu(:,i-1,j)'*d_r*Mu - (5/d^2)*(Mu'*d_r)*(mu(:,i-1,j)'*d_r)*d_r) ;
               B(:,i-1,j) = (mu_0/(4*pi*d^5))*(3*mu(:,i-1,j)'*d_r*d_r - d^2*mu(:,i-1,j));
            end
        else
            mu(1:3,i-1,j) = zeros(3,1);
            value_mu(i-1,j) = norm(mu(:,i-1,j));
            force(:,i-1,j) = zeros(3,1);
            B(:,i-1,j) = zeros(3,1);
        end
%         Mu = [7.7*1e22;0;0];
%         Rad_r =[Rad;0;0];
%         force_earth(:,i) = (3*mu_0/(4*pi*Rad^5))*(mu(i-1,:)*Mu*Rad_r + Mu'*Rad_r*mu(i-1,:)' + mu(i-1,:)*Rad_r*Mu - (5/Rad^2)*(Mu'*Rad_r)*(mu(i-1,:)*Rad_r)*Rad_r) ;
%         force(:,i) = force_earth(:,i);
%         Force(:,i) = Force(:,i) - force(:,i,j);
      
%         for k = 1:length(value_mu(i-1,:))
%             if k~=j
%                force_disturb{j,k}(:,i) = (3*mu_0/(4*pi*d^5))*(mu(:,i-1,j)'*(mu(:,i-1,k)*d_r + (mu(:,i-1,k)'*d_r*mu(:,i-1,j) + mu(:,i-1,j)'*d_r*(mu(:,i-1,k) - (5/d^2)*((mu(:,i-1,k)'*d_r)*(mu(:,i-1,j)'*d_r)*d_r) ;
%             end
%         end
    end     
%     [~,X_new] = ode45(@(t,X) Right_part(t,X,A'*Force(:,i)/M),[0:1:dt],X_1(:,i-1)); % Integration of the second satellite motion equations
%     X_1(:,i) = X_new(end,:)';
    for j=1:N
        force_disturb_sum(:,i-1,j) = zeros(3,1);
%         for k=1:N
%             if k~=j
%                d_r=xsat_diff(1:3,i-1,k)-xsat_diff(1:3,i-1,j);
%                d = norm(d_r); % distance between sats
%                reletive_R(i-1,j,k) = d;
%                d_critical = 0.05;
%                if d > d_critical
%                   force_disturb{k,j}(:,i-1) = (3*mu_0/(4*pi*d^5))*(mu(:,i-1,k)'*mu(:,i-1,j)*d_r + mu(:,i-1,j)'*d_r*mu(:,i-1,k) + mu(:,i-1,k)'*d_r*mu(:,i-1,j) - (5/d^2)*(mu(:,i-1,j)'*d_r)*(mu(:,i-1,k)'*d_r)*d_r) ;
%                   force_disturb_sum(:,i-1,j)=force_disturb_sum(:,i-1,j)+force_disturb{k,j}(:,i-1);
%                elseif norm(mu(:,i-1,k))~=0 && d <= d_critical
%                   mu(:,i-1,j)=zeros(3,1);
%                   force(:,i-1,j)=zeros(3,1);
%                   force_disturb_sum(:,i-1,j) = zeros(3,1);
%                end
%             end
%         end
               
        Force(:,i-1) = Force(:,i-1) + force(:,i-1,j);
        B_sum(:,i-1) = B_sum(:,i-1) + B(:,i-1,j);
        force_sum(:,i-1,j) = force(:,i-1,j)+force_disturb_sum(:,i-1,j);
        options.h = 0.05;
        [~,X_new] = ode4(@(t,X) Right_part_angular(t,X,mu(1:3,i-1,j),0,J_ChipSat,A'*(force(:,i-1,j)+ force_disturb_sum(:,i-1,j))/m),[0,dt],X_k(:,i-1,j),options); % Integration of the satellite motion equations
        X_k(:,i,j) = X_new(end,:)';
        w_ang_k(:,i,j) = X_k(11:13,i,j);
        qr_k(:,i,j) = X_k(7:10,i,j);
        [ang1_k(i,j), ang2_k(i,j), ang3_k(i,j)] = quat2angle(qr_k(:,i,j)');
    end
    
    if norm(C_1(1:procent*N)) < 0.05
       C_1(1)
       options.h = 0.05;
       [~,X_new] = ode4(@(t,X) Right_part_BDot(t,X,B_sum(:,i-1),A'*Force(:,i-1)/M),[0,dt],X_1(:,i-1),options); % Integration of the second satellite motion equations
    else
        options.h = 0.05;
        [~,X_new] = ode4(@(t,X) Right_part_angular(t,X,Mu,B_sum(:,i-1),J_3U,A'*Force(:,i-1)/M),[0,dt],X_1(:,i-1),options); % Integration of the second satellite motion equations  
    end
    X_1(:,i) = X_new(end,:)';
    w_ang(:,i) = X_1(11:13,i);
    qr(:,i) = X_1(7:10,i);
    [ang1(i), ang2(i), ang3(i)] = quat2angle(qr(:,i)');
%     
%     [~,X_new] = ode45(@(t,X) Right_part(t,X,A'*Force(:,i-1)/M),[0:1:dt],X_1(1:6,i-1)); % Integration of the second satellite motion equations
%     X_1(1:6,i) = X_new(end,:)';
    
    Omega(:,i) = cross(X_1(1:3,i),X_1(4:6,i))/norm(X_1(1:3,i))^2;% Orbital angular velocity
    A = orbital_dcm(X_1(1:6,i)); % Transition matrix to orbital reference frame

    for j=1:N        
        xsat_diff(:,i,j) = [A*(X_k(1:3,i,j) - X_1(1:3,i));A*((X_k(4:6,i,j) - X_1(4:6,i))-cross(Omega(:,i),(X_k(1:3,i,j) - X_1(1:3,i))))]; % Relative state vector in orbital reference frame
    end
end

k = 0;
for j = 1:N
    if abs(C_constants(1,end,j)) < 0.05
       k = k+1;
    end
end
out_amount = k/N;
 
figure
for k=1:N
    plot3(xsat_diff(1,:,k),xsat_diff(2,:,k),xsat_diff(3,:,k),'LineWidth',1)
    colormap(hsv(2*k+5))
    hold on
end
grid on
axis equal
xlabel('x, m') 
ylabel('y, m')
zlabel('z, m')

figure
for k=1:N
    plot(xsat_diff(1,:,k),xsat_diff(3,:,k),'LineWidth',1)
    colormap(hsv(2*k+5))
    hold on
end
grid on
% axis equal
xlabel('x, m') 
ylabel('z, m')

% mu_2_max = zeros(length(value_mu(:,1)));
% mu_2_max(:) = mu_max_value;
% figure
% for k=1:N
%     plot(value_mu(:,k))
%     colormap(hsv(2*k+5))
%     plot(mu_2_max,'LineWidth',2)
%     hold on
% end
% grid on
% % axis equal
% xlabel('time, s') 
% ylabel('magnetic moments, Am^2')


% for j = 1:N
%     figure
%     for i=1:3
%         plot(mu(i,:,j))
%         hold on
%     end
%     % legend('mu_x','mu_y','mu_z')
%     xlabel('Time, s') 
%     ylabel('Components of magnetic moment, Am^2')
% end

figure
for i=1:N
    plot(C_constants(1,:,i))
    hold on
end
xlabel('Time, s') 
ylabel('C1 relative drift, m')

figure
for i=1:N
    plot(value_mu(:,j))
    hold on
end
xlabel('Time, s') 
ylabel('Magnetic moments of ChipSats, Am^2')

% figure
% for i=1:N
%     plot(mu(1,:,i))
%     hold on
% end
% % legend('mu_x','mu_y','mu_z')
% xlabel('time, s') 
% ylabel('oX component of magnetic moments, Am^2')
% 
% figure
% plot(T,force(1,:),T,force(2,:),T,force(3,:),'LineWidth',1)
% grid on
% % axis equal
% xlabel('x, m') 
% ylabel('z, m')

% for i=1:N
%     figure
%     plot(T,force_sum(1,:,i),T,force_sum(2,:,i),T,force_sum(3,:,i),'LineWidth',1)
%     xlabel('time, s') 
%     ylabel('components of force, N')
% end
figure
hold on
plot(qr(1, :), 'k');
plot(qr(2, :), 'r');
plot(qr(3, :), 'g');
plot(qr(4, :), 'b');
xlabel('Time, s')
ylabel('Quaternion components')
legend('q_1','q_2','q_3','q_4')
grid on

figure
hold on
plot(ang1*180/pi, '-r');
plot(ang2*180/pi, '-g');
plot(ang3*180/pi, '-b');
xlabel('Time, s')
ylabel('Attitude angles, deg')
grid on
legend('\alpha','\beta','\gamma')

figure
hold on
plot(w_ang(1,:)*180/pi, '-r');
plot(w_ang(2,:)*180/pi, '-g');
plot(w_ang(3,:)*180/pi, '-b');
xlabel('Time, s')
ylabel('Angular velocity, deg/s')
grid on
legend('\omega_x','\omega_y','\omega_z')

% 
% for i=1:4
%     figure
%     for k = 1:N
%         hold on
%         plot(qr_k(i, :, k));
%     end 
% end
% xlabel('�����, �')
% ylabel('quaternion')
% grid on
% 
% for k = 1:N
%     figure
%     hold on
%     plot(ang1_k(:,k)*180/pi, '-r');
%     plot(ang2_k(:,k)*180/pi, '-g');
%     plot(ang3_k(:,k)*180/pi, '-b');
% end
% xlabel('�����, �')
% ylabel('angles')
% grid on
% legend('\alpha','\beta','\gamma')
% 
% for k = 1:N
%     figure
%     for i = 1:3
%     hold on
%     plot(w_ang_k(i,:,k)*180/pi, '-r');
%     end 
% end
% xlabel('�����, �')
% ylabel('angular velocity')
% grid on
% legend('\omega_x','\omega_y','\omega_z')