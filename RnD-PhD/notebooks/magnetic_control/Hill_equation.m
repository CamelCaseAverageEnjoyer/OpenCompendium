function dy = Hill_equation(~,y,W)
dy = zeros(6,1);    % a column vector
dy(1) = y(4);
dy(2) = y(5);
dy(3) = y(6);
dy(4) = -2*W*y(6);
dy(5) = -(W^2)*y(2);
dy(6) = 2*W*y(4) + 3*(W^2)*y(3);
end