function show_marker(r, r1, q, d)
    marker = [1 1 1 0;
              1 0 1 1;
              1 1 0 1;
              1 1 1 1];
    marker_clr = [0; 1];
    n = size(marker);
    n = n(1);
    h = 1 / (n + 2);
    
    x = zeros(4, n^2 + 4);
    y = zeros(4, n^2 + 4);
    z = zeros(4, n^2 + 4);
    c = zeros(1, n^2 + 4);
    
    for i = 1:n
        for j = 1:n
            x(:, n*i+j) = [h*i; h*i+h; h*i+h; h*i];
            y(:, n*i+j) = [h*j; h*j; h*j+h; h*j+h];
            z(:, n*i+j) = [0; 0; 0; 0];
            c(n*i+j) = marker_clr(marker(i,j) + 1);
        end
    end
    
    % ??????????????k?? b????????k????
    x(:, n*i+j+1:n*i+j+4) = [0 0 (1-h) 0; h 0 1 0; h 1 1 1; 0 1 (1-h) 1];
    y(:, n*i+j+1:n*i+j+4) = [0 0 0 (1-h); 0 h 0 1; 1 h 1 1; 1 0 1 (1-h)];
    z(:, n*i+j+1:n*i+j+4) = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];
    c(n*i+j+1:n*i+j+4) = [0; 0; 0; 0];
    x = x - 0.5;
    y = y - 0.5;

    % Y????l????????????
    x = x * d;
    y = y * d;

    % ????????ll??l??????j ??????????????
    x = x + r1(1);
    y = y + r1(2);
    z = z + r1(3);

    % ????????????????

    % ????????ll??l??????j ??????????????
    x = x + r(1);
    y = y + r(2);
    z = z + r(3);

    patch(x,y,z,c);
end