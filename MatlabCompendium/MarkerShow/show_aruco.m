function show_aruco(dims, A, r_irf, r_brf)  % (r, r1, q, d)
    % Marker define
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
    
    % Aruco borders
    x(:, n*i+j+1:n*i+j+4) = [0 0 (1-h) 0; h 0 1 0; h 1 1 1; 0 1 (1-h) 1];
    y(:, n*i+j+1:n*i+j+4) = [0 0 0 (1-h); 0 h 0 1; 1 h 1 1; 1 0 1 (1-h)];
    z(:, n*i+j+1:n*i+j+4) = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];
    c(n*i+j+1:n*i+j+4) = [0; 0; 0; 0];
    x = x - 0.5;
    y = y - 0.5;

    % BRF -> IRF
    x = reshape(x, 1, []);
    y = reshape(y, 1, []);
    z = reshape(z, 1, []);
    for i = 1:length(x)
        r = [x(i);y(i);z(i)];
        r = r .* dims;  % stretching
        r = r + r_brf;  % translation
        r = A * r;  % rotation
        r = r + r_irf;  % translation
        x(i) = r(1);
        y(i) = r(2);
        z(i) = r(3);
    end
    x = reshape(x, 4, []);
    y = reshape(y, 4, []);
    z = reshape(z, 4, []);

    patch(x,y,z,c);
end


