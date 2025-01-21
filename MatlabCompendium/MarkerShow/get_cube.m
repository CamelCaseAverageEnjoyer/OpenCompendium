function [x,y,z] = get_cube(dims, A, tns)  % A, D
    % Arguments
    % A [mat] rotation matrix
    % dims [vec] stretching vector
    % tns [vec] translation vector
    h = 0.5;
    r = [-h -h -h h -h -h h h -h h h h h h -h h h h -h -h -h h -h -h;
         -h -h -h -h -h h -h -h h h -h h h h h h -h h h h -h -h -h h;
         -h h -h -h -h -h -h h -h -h -h -h -h h h h h h -h h h h h h];
    for i = 1:length(r(1,:))
        r(:,i) = r(:,i) .* dims;  % stretching
        r(:,i) = A * r(:,i);  % rotation
        r(:,i) = r(:,i) + tns;  % translation
    end
    x = reshape(r(1,:),[6,4])';
    y = reshape(r(2,:),[6,4])';
    z = reshape(r(3,:),[6,4])';
end


