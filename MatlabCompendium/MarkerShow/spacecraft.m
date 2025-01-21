classdef spacecraft
    properties
        r_irf
        r_orf
        v_orf
        q_irf
        w_irf
        w_orf
        cam_pos = [0;0;0]
        cam_dir = [0;0;0]
        cam_up = [0;0;0]
    end

    methods
        function self = spacecraft(d, r_orf, v_orf, q_irf, w_orf)
            self.v_orf = v_orf;
            self.r_orf = r_orf;
            self.r_irf = d.o2i_r(r_orf);
            self.q_irf = q_irf;
            self.w_orf = w_orf;
            self.w_irf = d.o2i_w(w_orf);
        end

        function anw = get_campos_irf(self)
            A = q2dcm(self.q_irf);
            anw = A * self.cam_pos + self.r_irf;
        end
        function anw = get_camdir_irf(self)
            A = q2dcm(self.q_irf);
            anw = A * self.cam_dir;
        end
        function anw = get_camup_irf(self)
            A = q2dcm(self.q_irf);
            anw = A * self.cam_up;
        end

        function show_chief(self)
            % Chief spacecraft is box
            dims = [0.3; 0.5; 0.3];
            A = q2dcm(self.q_irf);
            [x, y, z] = get_cube(dims, A, self.r_irf);        
            patch(x,y,z,0.5);
        end

        function show_deputy(self)
            % Corpus show (box)
            dims = [0.1; 0.1; 0.01];
            A = q2dcm(self.q_irf);
            [x, y, z] = get_cube(dims, A, self.r_irf);        
            patch(x,y,z,0.5);

            % Aruco show
            dim1 = 0.03;
            r_brf = [dims(1)/2 - dim1/2; % Its position on deputy spacecraft
                     dims(2)/2 - dim1/2; 
                     dims(3)/2 + 0.003];
            % show_marker(r, [d(1)/2-d1/2; d(2)/2-d1/2; d(3)/2+0.0001], q, d1);  
            show_aruco(dim1, A, self.r_irf, r_brf);  
        end
    end
end
