function [xsat_est, P] = Kalman_rel_est(xsat_est, P, Measurements, dt, D, R, w, control)

% Kalman filter
% Parameters:
% xsat_est - Current state estimation
% P - Covariance matrix
% Measurements - Current measurements
% dt - sampling matrix
% D - disturbance covariance matrix 
% R - measurement covariance matrix
% Control - control of the satellite position
% w - orbital angular velocity

%% Motion equation integration

xsat_est = RK4FK(xsat_est,w, dt,control);


%% Dinamical matrix

F = [0 0    0     0    0 -2*w;
     0 -w^2 0     0    0 0;
     0 0    3*w^2 2*w  0 0];

F = [zeros(3) eye(3);
     F ];
 
Fi = eye(6) + F*dt;     % State transition matrix

B = [zeros(3); eye(3)]; % Control matrix
Q = Fi*B*D*B'*Fi'*dt;   % Dynamical noise matrix
H = [eye(3) zeros(3)];  % Measurement matrix

%% Covariance matrix calculation
P=Fi*P*Fi'+Q;
K=P*H'*inv(H*P*H'+R);    


%% State vector correction
if norm(Measurements)~=0
    xsat_est = xsat_est + K*(Measurements-xsat_est(1:3));
    P=P-K*H*P;
end

end
