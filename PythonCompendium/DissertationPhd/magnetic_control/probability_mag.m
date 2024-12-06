clc
close all
clear all

N = 16; %количество спутников
number = 50; %общее количество запусков для одного t_Fire
V0_launch(1) = (rand(N/4,1)+0.25)*1e-2;
V0_launch(2) = (rand(N/4,1)+0.5)*1e-2;
V0_launch(3) = (rand(N/4,1)+0.75)*1e-2;
V0_launch(4) = (rand(N/4,1)+1)*1e-2;
V0_launch(5) = (rand(N/4,1)+1.25)*1e-2;
t_end = 4*60*90;
var1 = zeros(number,length(V0_launch));

for i = 1:1:length(V0_launch)

    for count = 1:number
        i 
        count
        tic 
       out_variation1 = Mag_Control_func(V0_launch(i), N, T_end);

    toc
    var1(count,i) = out_variation1;

    end 
end

save('Magnetic control v0 launch.mat')
% X = t0*10:dt*10:tend*10;
boxplot(var1,V0_launch);
ylim([0 1])
xlabel('V_0 _l_a_u_n_c_h')
ylabel('N  _c_l_u_s_t_e_r  / N  _t_o_t_a_l')
