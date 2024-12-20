%% Measurement noise parameters
    sigma_meas = [0.5;0.1*pi/180;0.1*pi/180]; % noise of the lazer range finder, and angles measurement from optical sensor
%% Kalman filter parameters
    sigma_d = 5e-5;                                        %Diagonal element of the matrix D
    sigma_r0 = 1;                                          %Diagonal element of the matrix P0
    sigma_v0 = 0.1;                                         %Diagonal element of the matrix P0
    Kalman_D = diag([ones(3,1)*sigma_d^2]);                %Matrix D
    P = diag([ones(3,1)*sigma_r0^2;ones(3,1)*sigma_v0^2]); %Matrix P0
    xsat_est(4:6,1) = normrnd(0,sigma_v0,3,1);             %Relative velocity errors