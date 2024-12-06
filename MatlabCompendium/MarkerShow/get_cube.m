function [x,y,z] = get_cube()
    h = 0.5;
    x = [-h -h -h h -h -h; h h -h h h h; h h -h h h h; -h -h -h h -h -h];
    y = [-h -h -h -h -h h; -h -h h h -h h; h h h h -h h; h h -h -h -h h];
    z = [-h h -h -h -h -h; -h h -h -h -h -h; -h h h h h h; -h h h h h h];
end

