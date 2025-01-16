from datetime import datetime
import numpy as np

# Писать следующим образом:
# ---> в _control.py

def Mag_Control_func(speed_v0, N, T_end):
    from _tiny_functions import orbitalMotionKeplerian
    """
    :param speed_v0:
    :param N:
    :param T_end:
    :return: out_amount
    """
    
    dt = 10  # Запихнуть в глобальные переменные? Шаг сипуляции
    T = np.linalg.norm(0, dt, T_end)
    
    # X_1 = zeros(6,length(T)); % First Satellite position in Earth-centered inertial reference frame
    # X_2 = zeros(6,length(T)); % Second Satellite position in Earth-centered inertial reference frame
    # xsat_diff_12 = zeros(6,length(T)); % Relative state vector
    mu = 3.986e14  # Гравитационный параметр
    R_earth = 6.371e6
    
    # Параметры спутника-преследователя
    Hight = 500e3  # Высота спутника
    Rad = R_earth + Hight  # Высота орбиты
    incl_1 = 51.7*np.pi/180  # Наклонение
    epsilon = 0  # Экцентриситет
    phi = 0  # Долгота восходящего узла
    omega = 0  # Аргумент перицентра
    t_pi = 0  # Перицентровое время
    t_cur = 0  # Текущее время
    approx = 0  # Первый шаг для метода Ньютона
    r, v, D = orbitalMotionKeplerian(mu, Rad, epsilon, phi, omega, incl_1, t_pi, t_cur, approx);
    
    m = 4e-3  # Масса ChipSat
    M = 3  # Масса материнского КА
    mu_0 = 4*np.pi*1e-7  # Проницаемость свободного пространства
    Mu = np.array([0.5, 0.5, 0.5])  # Магнитный момент материнского КА
    
    X_1 = np.append(r, v)  # Абсолютный вектор начального состояния спутника
    
    Omega(:,1) = cross(X_1(1:3,1),X_1(4:6,1))/norm(X_1(1:3,1))^2; % Orbital angular velocity
    A = orbital_dcm(X_1(:,1)); % Transition matrix to orbital reference frame
    
    %initial angular state
    qr0=angle2quat(10.0*pi/180,10.0*pi/180,10.0*pi/180,'YZX')';
    X_1(7:10,1) = qr0;
    X_1(11:13,1) = Omega(:,1);
    
    # Параметры других спутников
    R0_launch = zeros(N,3);
    V0_launch(1:N,1:3) = (rand(N,3)-0.25)*1e-3; %0.5*1e-3*ones(N,3);
    V0_launch(1:N/4,1) = speed_v0; %0.5*1e-2*ones(N/4,1); %(rand(N/4,1)+0.5)*1e-2; %velocity ox positive direction
    V0_launch(N/4+1:N/2,3) = speed_v0; %0.5*1e-3*ones(N/4,1); %(rand(N/4,1)+0.5)*1e-3; %velocity oz positive direction
    V0_launch(N/2+1:3*N/4,1) = -speed_v0; %;%-(rand(N/4,1)+0.5)*1e-2; %velocity ox negative direction
    V0_launch(3*N/4+1:N,3) = -speed_v0;%-0.5*1e-3*ones(N/4,1);%-(rand(N/4,1)+0.5)*1e-3; %velocity oz negative direction
    
    RV0_launch = [R0_launch, V0_launch];
    dt_launch = 10; %time between launches
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
    
    %% Main cycle
    for i = 2:1:length(T) 
        i
        Force(:,i-1) = zeros(3,1);
        B_sum(:,i-1) = zeros(3,1);
        for j = 1:N
            
            C_constants(:,i-1,j) = coord2const(xsat_diff(:,i-1,j), norm(Omega(:,i-1))); % Relative motion constants calculation 
            if abs(C_constants(1,i-1,j)) > 1e-5
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
                mu(:,i-1,j) = zeros(3,1);
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
                   
            Force(:,i-1) = Force(:,i-1) - force(:,i-1,j);
            B_sum(:,i-1) = B_sum(:,i-1) + B(:,i-1,j);
            force_sum(:,i-1,j) = force(:,i-1,j)+force_disturb_sum(:,i-1,j);
            [~,X_new] = ode45(@(t,X) Right_part(t,X,A'*(force(:,i-1,j)+ force_disturb_sum(:,i-1,j))/m),[0:1:dt-1],X_k(:,i-1,j)); % Integration of the satellite motion equations
            X_k(:,i,j) = X_new(end,:)';
        end
        
        [~,X_new] = ode4(@(t,X) Right_part_angular(t,X,Mu,B_sum(:,i-1),A'*Force(:,i-1)/M),[0:1:dt-1],X_1(:,i-1)); % Integration of the second satellite motion equations
        X_1(:,i) = X_new(:,end);
    %     Omega = cross(X_1(1:3,i),X_1(4:6,i))/norm(X_1(1:3,i))^2; % Orbital angular velocity
        Omega(:,i) = X_1(11:13,i);
        qr = X_1(7:10,i);
        [ang1(i) ang2(i) ang3(i)] = quat2angle(qr');
        A = orbital_dcm(X_1(:,i)); % Transition matrix to orbital reference frame
    
        for j=1:N        
            xsat_diff(:,i,j) = [A*(X_k(1:3,i,j) - X_1(1:3,i));A*((X_k(4:6,i,j) - X_1(4:6,i))-cross(Omega(:,i),(X_k(1:3,i,j) - X_1(1:3,i))))]; % Relative state vector in orbital reference frame
        end
    end
    k = 0;
    for j = 1:N
        if abs(C_constants(1,end,j)) < 0.1
           k = k+1;
        end
    end
    out_amount = k/N;
    
    return out_amount