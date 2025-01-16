syms mu1_x mu2_x mu2_y mu2_z dx dy dz koef F D

S = solve(koef*((mu1_x*mu2_x)*dx + (mu1_x*dx)*mu2_x + (mu2_x*dx+mu2_y*dy+mu2_z*dz)*mu1_x - (5/D^2)*(mu1_x*dx)*(mu2_x*dx+mu2_y*dy+mu2_z*dz)*dx)-F == 0,...
                              (mu1_x*mu2_x)*dy + (mu1_x*dx)*mu2_y - (5/D^2)*(mu1_x*dx)*(mu2_x*dx+mu2_y*dy+mu2_z*dz)*dy == 0,...
                              (mu1_x*mu2_x)*dz + (mu1_x*dx)*mu2_z - (5/D^2)*(mu1_x*dx)*(mu2_x*dx+mu2_y*dy+mu2_z*dz)*dz == 0,[mu2_x mu2_y mu2_z]);
S.mu2_x
S.mu2_y
S.mu2_z