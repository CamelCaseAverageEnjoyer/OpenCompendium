marker = [0 0 1 0;
          0 1 0 0;
          1 0 0 1;
          0 0 0 1];
marker_clr = [[0 0 0]; [1 1 1]];
n = size(marker);
n = n(1);

x = zeros(4, n^2);
y = zeros(4, n^2);
z = zeros(4, n^2);
c = zeros(1, n^2);

for i = 1:n
    for j = 1:n
        t = 0.01;
        h = 1/n;
%         v = [h*i+t, h*j+t, 0; 
%              h*i+h-t, h*j+t, 0; 
%              h*i+h-t, h*j+h-t, 0; 
%              h*i+t, h*j+h-t, 0]; 
%         fac = [1 2 3 4]; 
%         patch('Vertices',v,'Faces',fac,'Color', 'green') 

        x(:, n*i+j) = [h*i+t; h*i+h-t; h*i+h-t; h*i+t];
        y(:, n*i+j) = [h*j+t; h*j+t; h*j+h-t; h*j+h-t];
        z(:, n*i+j) = [0; 0; 0; 0];
        c(n*i+j) = marker_clr(marker(i,j) + 1, 1);
    end
end

patch(x,y,z,c);
