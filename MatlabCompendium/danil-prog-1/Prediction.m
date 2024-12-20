function [xsat, P] = Prediction(xsat_est, P, dt, T, D, w)

% Kalman filter
% Parameters:
% xsat_est - Current state estimation
% P - Covariance matrix
% Measurements - Current measurements
% dt - sampling time
% Time of prediction
% D - disturbance covariance matrix 
% Control - control of the satellite position
% w - orbital angular velocity
xsat = zeros(6,length([0:dt:T]));
xsat(:,1) = xsat_est;



%% Dinamical matrix
F = [0 0    0     0    0 -2*w;
     0 -w^2 0     0    0 0;
     0 0    3*w^2 2*w  0 0];

F = [zeros(3) eye(3);
     F ];
 
Fi = eye(6) + F*dt;     % State transition matrix

B = [zeros(3); eye(3)]; % Control matrix
Q = Fi*B*D*B'*Fi'*dt;   % Dynamical noise matrix

control = zeros(3,1);   % Passive motion
%% Prediction
for i = 2:length(xsat(1,:));
    
    xsat(:,i) = RK4FK(xsat(:,i-1),w, dt,control);
    P = Fi*P*Fi'+Q;

end

% P
