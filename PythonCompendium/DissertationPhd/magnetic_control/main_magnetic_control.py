import numpy as np

def simulation_magnetic_control(N: int = 8,
                                dt: float = 10,
                                T_end: float = 4*60*90,
                                k_control: float = 10000) -> None:
    """
    :param N: Количество аппаратов
    :param dt: Шаг по времени
    :param T_end: Время моделирования
    :param k_control: Для BDot"""

    # Параметры численного моделирования
    T = np.linspace(0, T_end, int(T_end // dt))
    mu = 3.986e14  # Гравитационный параметр
    R_earth = 6.371e6  # Радиус Земли
    mu_0 = np.pi*4e-7; # Permeability of free space
    Mu_earth = 7.72*1e22*np.array([0, 0, -1])  # A*m2 direct dipole for Earth

    # Параметры аппаратов
    m = 4*1e-3  # Масса ChipSat, кг
    M = 3  # Масса CubeSat, кг
    Mu = np.array([0.5, 0.5, 0.5])  # Магнитный момент материнского (мэйн) аппарата 
    mu_max_value = 0.005  # Максимальный магнитный момент ChipSats
    J_3U = np.diag([0.005, 0.025, 0.025])  # Тензор инерции ChipSat
    J_ChipSat = np.diag([2.5e-7, 1.25e-7, 1.25e-7])  # Тензор инерции CubeSat

    # Начальные параметры CubeSat и ChipSat 
    Hight = 500e3  # Высота орбиты
    Rad = R_earth + Hight
    incl_1 = 51.7*np.pi/180  # Наклонение
    epsilon = 0  # Эксцентриситет
    phi = 0  # Долгота восходящего узла
    omega = 0  # Аргумент перицентра
    t_pi = 0  # Время перицентра
    t_cur = 0  # Текущее время
    approx = 0  # Первый шаг для Метода Ньютона
    r, v, D = orbitalMotionKeplerian(mu=mu, Rad=Rad, epsilon=epsilon, phi=phi, omega=omega, incl_1=incl_1, t_pi=t_pi, t_cur=t_cur, approx=approx)

    X_1 = np.append(r, v)  # Абсолютный(ИСК?) НУ вектор CubeSat

    Omega = my_cross(X_1[0:3], X_1[3:6]) / np.linalg.norm(X_1[0:3])**2  # Орбитальная угловая скорость
    # Omega(:,1) = cross(X_1(1:3,1),X_1(4:6,1)) / norm(X_1(1:3,1))^2; % Orbital angular velocity
    A = orbital_dcm(X_1)  # Матрица перехода в ОСК

    # Вращательные НУ CubeSat
    ang1 = 5.0*np.pi/180  # Углы в радианах
    ang2 = 8.0*np.pi/180
    ang3 = 3.0*np.pi/180
    qr = angle2quat(ang=[ang1, ang2, ang3], 'YZX')
    w_ang = np.array([1.0]*3)*np.pi/180
    X_1 = np.append(X_1, qr)
    X_1 = np.append(X_1, w_ang)

# ChipSats скорости запуска
R0_launch = np.zeros(N,3)
V0_launch(1:N,1:3) = np.rand(N,3)*1e-3 - 0.25*1e-3*np.ones(N,3)  # 0.5*1e-3*ones(N,3)
V0_launch(1:N/4,1) = (rand(N/4,1)*1e-3) + (0.5*1e-2*ones(N/4,1))  # 0.5*1e-2*ones(N/4,1)  # (rand(N/4,1)+0.5)*1e-2  # velocity ox positive direction
V0_launch(N/4+1:N/2,3) = (rand(N/4,1)*1e-3) + (0.5*1e-2*ones(N/4,1))  # 0.5*1e-3*ones(N/4,1)  # (rand(N/4,1)+0.5)*1e-3  # velocity oz positive direction
V0_launch(N/2+1:3*N/4,1) = -(rand(N/4,1)*1e-3) - (0.5*1e-2*ones(N/4,1))  # -(rand(N/4,1)+0.5)*1e-2  # velocity ox negative direction
V0_launch(3*N/4+1:N,3) = -(rand(N/4,1)*1e-3) - (0.5*1e-2*ones(N/4,1))  # -0.5*1e-3*ones(N/4,1)  # -(rand(N/4,1)+0.5)*1e-3  # velocity oz negative direction

    RV0_launch = np.htrack(R0_launch, V0_launch)  # МОЖЕТ, ПОПРАВИТЬ
    dt_launch = 5.  # Время между запусками
    tend_launch = N / 4 * dt_launch  # Время когда начинается управление  А КАКОГО ХУЯ ПОДЕЛИТЬ НА 4 ТО (ШАМАН)
    R0 = np.zeros(N,3)
    V0 = np.zeros(N,3)

    for j in range(int(N // 4)):
        for k=j:N/4:N np.linspace() # ЭТО БЛЯТЬ КАК ВООБЩЕ НАХУЙ (ШАМАН) было j:N/4:N
            k = int(k)
            y0 = RV0_launch(k,:)
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

%initial angular state for ChipSats
for j = 1:N
    w_ang_k(:,1,j) = w_ang(:,1);
    qr_k(:,1,j) = qr(:,1);
    ang1_k(1,j) = ang1(1);
    ang2_k(1,j) = ang2(1); 
    ang3_k(1,j) = ang3(1);
    X_k(7:10,1,j) = qr_k(:,1,j);
    X_k(11:13,1,j) = w_ang_k(:,1,j);
end
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
    for j = 1:N
        C_constants(:,i-1,j) = coord2const(xsat_diff(:,i-1,j), norm(Omega(:,i-1))); % Relative motion constants calculation 
        force_desired(i-1,j) = -m*C_constants(1,i-1,j)*norm(Omega(:,i-1))/(2*dt); % desired force
        % counts required magnetic moment for translational motion
        [MU] = magnetic_dipole(force_desired(i-1,j), xsat_diff(1:3,i-1,j), Mu); %in OrbRF
        mu(1:3,i-1,j) = MU;

        [B_3U] = mag_field(mu_0, xsat_diff(1:3,i-1,j), Mu); %in OrbRF
        [B_earth] = mag_field(mu_0, X_k(1:3,i-1,j), Mu_earth); %in earth IRF
        B(1:3,i-1,j) = A'*B_3U + B_earth; %mag field in earth IRF    
        D = quat2dcm(X_k(7:10,i-1,j)'); % to BF
%         mu_BDot(1:3,i-1,j) = k_control*cross(w_ang_k(:,i-1,j),D*B(1:3,j)); % to BF 
        mu_BDot(1:3,i-1,j) = zeros(3,1);
        mu_sum(1:3,i-1,j) = D*(A'*mu(:,i-1,j)) + mu_BDot(:,i-1,j); % in BF
        value_mu(i-1,j) = norm(mu_sum(:,i-1,j));
        if value_mu(i-1,j) > mu_max_value
           mu_sum(:,i-1,j) = mu_max_value*(mu_sum(:,i-1,j)/norm(mu_sum(:,i-1,j)));
           value_mu(i-1,j) = norm(mu_sum(:,i-1,j));
        end
    end   
    
    Force(:,i-1) = zeros(3,1);
    B_sum(:,i-1) = zeros(3,1);
    for j=1:N
        force_disturb_sum(:,i-1,j) = zeros(3,1);
        B_disturb_sum(:,i-1,j) = zeros(3,1);
%         for k = 1:N
%             if k~=j
%                d = xsat_diff(1:3,i-1,j)-xsat_diff(1:3,i-1,k);
%                [force_disturb(1:3,k,j)] = dipole_force(mu_0,d,A*(D'*mu_sum(:,i-1,j)),A*(D'*mu_sum(:,i-1,k)));%in OrbRF
%                force_disturb_sum(:,i-1,j)=force_disturb_sum(:,i-1,j)+force_disturb(:,k,j);%in OrbRF
%                [B_disturb(1:3,k,j)] = mag_field(mu_0,d,A*(D'*mu_sum(:,i-1,k)));%in OrbRF
%                B_disturb_sum(:,i-1,j) = B_disturb_sum(:,i-1,j)+B_disturb(:,k,j);%in OrbRF
%             end
%         end
        [force_3U] = dipole_force(mu_0,xsat_diff(1:3,i-1,j),A*(D'*mu_sum(:,i-1,j)),Mu); %in OrbRF on Chipsat j
        [force_earth] = dipole_force(mu_0,X_k(1:3,i-1,j),D'*mu_sum(:,i-1,j),Mu_earth); %in earth IRF on Chipsat j
        force_sum(1:3,i-1,j) = A'*force_3U; %+ force_earth; %in earth IRF on Chipsat j
        Force(:,i-1) = Force(:,i-1) - force_3U; %acts on cubesat from chipsats in OrbRF
        [B_on_3U] = mag_field(mu_0,-xsat_diff(1:3,i-1,j),A*(D'*mu_sum(:,i-1,j)));%in OrbRF acts on cubesat from chipsat
        B_sum(:,i-1) = B_sum(:,i-1) + B_on_3U; %acts on cubesat from all chipsats in OrbRF
        options.h = 0.05;
        %Right_part(t,X,Mu,B,J,F)
        [~,X_new] = ode4(@(t,X) Right_part(t,X,mu_sum(:,i-1,j),B(1:3,i-1,j)+A'*B_disturb_sum(:,i-1,j),J_ChipSat,(force_sum(1:3,i-1,j)+A'*force_disturb_sum(:,i-1,j))/m),[0,dt],X_k(:,i-1,j),options); % Integration of Chipsat motion equations
        X_k(:,i,j) = X_new(end,:)';
        w_ang_k(:,i,j) = X_k(11:13,i,j);
        qr_k(:,i,j) = X_k(7:10,i,j);
        [ang1_k(i,j), ang2_k(i,j), ang3_k(i,j)] = quat2angle(qr_k(:,i,j)');
    end
    [B_earth_on_3U] = mag_field(mu_0, X_1(1:3,i-1), Mu_earth); %in earth IRF acts on cubesat from earth
    B_on_3U_sum(:,i-1) = B_earth_on_3U + A'*B_sum(:,i-1); %in earth IRF
    [force_earth_on_3U] = dipole_force(mu_0,X_1(1:3,i-1),A'*Mu,Mu_earth); %in earth IRF on Cubesat
    F_on_3U_sum(:,i-1) = A'*Force(:,i-1); %+ force_earth_on_3U; %in earth IRF on Cubesat
    options.h = 0.05;
    %Right_part(t,X,Mu,B,J,F)
    [~,X_new] = ode4(@(t,X) Right_part(t,X,D*(A'*Mu),B_on_3U_sum(:,i-1),J_3U,F_on_3U_sum(:,i-1)/M),[0,dt],X_1(:,i-1),options); % Integration of the Cubesat motion equations
    X_1(:,i) = X_new(end,:)';
    w_ang(:,i) = X_1(11:13,i);
    qr(:,i) = X_1(7:10,i);
    [ang1(i), ang2(i), ang3(i)] = quat2angle(qr(:,i)');

    
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
    plot(value_mu(:,i))
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































