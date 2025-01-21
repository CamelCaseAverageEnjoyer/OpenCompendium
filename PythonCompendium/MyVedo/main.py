import numpy as np
from stl import mesh
import vedo
from vedo import dataurl, Mesh, Plotter
from vedo import IcoSphere, Earth, show
from vedo import dataurl, show, Cube

from get_cube import *
from dynamics import *

earth_rate = 1e5

def iteration_timer(eventId=None):
    global fig_view, camera

    d.time_step(spacecrafts)

    # Deploying
    for j in range(n_deputy):
        if (not if_deployed_deputy[j]) and d.t >= t_deploy_deputy[j]:
            if_deployed_deputy[j] = True
            spacecrafts.append(Spacecraft(d, r_orf_deputy[j], v_orf_deputy[j],
                                             q_irf_deputy[j], w_orf_deputy[j]))

    mesh = IcoSphere(pos=(d.r_orb, 0, 0), r=d.r_earth, subdivisions=5, alpha=1.0).texture('img/earth2.jpg') 
    mesh += c.show_chief(d)
    for j in range(2, len(spacecrafts)):
        mesh += spacecrafts[j].show_deputy(d)
    sun_point = vedo.Point(d.i2o_r([0,-1e5,0]), c='y')
    sun_light = vedo.Light(sun_point, c='white', intensity=3)
    mesh += sun_point
    mesh += sun_light
    fig_view.pop().add(mesh).render()

    camera.SetPosition(c.get_campos_orf(d))
    camera.SetEyePosition(c.get_camdir_orf(d))

def button_func(obj, ename):
    global timerId
    fig_view.timer_callback("destroy", timerId)
    if "Play" in button.status():
        timerId = fig_view.timer_callback("create", dt=100)
    button.switch()

    # timerId = fig_view.timer_callback("create", dt=100)

global timerId, fig_view, button, evnetId, camera

timerId = 1
fig_view = Plotter(bg='k', size=(900, 600))
# button = fig_view.add_button(button_func, states=["Play ", "Pause"], size=10, bold=True, pos=[0.98, 1])
fig_view.timer_callback("destroy", timerId)
timerId = fig_view.timer_callback("create", dt=100)
evnetId = fig_view.add_callback("timer", iteration_timer)

""" Типа документация
styles_of_light = ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']
"""

""" Динамика системы """
h_orb = 400e3
dt = 1.
t_modeling = 100.
d = Dynamics(h_orb=h_orb, dt=dt)

""" Материнский аппарат """
r_orf_chief = np.array([0, 0, 0])
v_orf_chief = np.array([0, 0, 0])
q_irf_chief = np.array([1, 0, 0, 0])
w_orf_chief = np.array([0, 0.01, 0])
c = Spacecraft(d,r_orf_chief,v_orf_chief,q_irf_chief,w_orf_chief)
c.cam_pos = np.array([0, 0, 0])
c.cam_dir = np.array([0, 4, -0.1])
c.cam_up =  np.array([0, 0, 1])
spacecrafts = [c]
chief_mesh = c.show_chief(d)  # .lighting('ambient')  # .color("silver")

""" Дочерние аппараты """
n_deputy = 3
t_deploy_deputy = [10, 20, 30]
if_deployed_deputy = [False, False, False]
r_orf_deputy = [np.array([0,0.1,0]) for i in range(3)]  # in m
v_orf_deputy = [np.array([0,0.01,0]) for i in range(3)]  # in m/s
i = 1/np.sqrt(2)
q_irf_deputy = [np.array([-i, i, 0, 0]) for i in range(3)]
w_orf_deputy = [np.zeros(3) for i in range(3)]


""" Земля """
d.r_orb /= earth_rate
d.r_earth /= earth_rate
earth_mesh = IcoSphere(pos=(d.r_orb, 0, 0), r=d.r_earth, subdivisions=4, alpha=1.0).texture('img/earth2.jpg')  # .color("silver") 
# earth = vedo.Earth(r=2, style=3)


""" Освещение """
sun_point = vedo.Point(d.i2o_r([0,-1e5,0]), c='y')
sun_light = vedo.Light(sun_point, c='white', intensity=3)

camera = vedo.oriented_camera(center=c.get_campos_orf(d),
                              up_vector=c.get_camup_orf(d),
                              backoff_vector=c.get_camdir_orf(d))

print(f"Camera: r={c.get_campos_orf(d)}, dir={c.get_camdir_orf(d)}, \nSpacecraft r={c.r_orf}")

fig_view.show(__doc__, earth_mesh + chief_mesh + sun_light + sun_point, camera=camera, resetcam=False, zoom=1, bg='k')  # , bg="hdri/2.hdr")


"""clear;

%% Params of display
window_size = [900 600];
camera_angle = 50;
is_animate = false;

%% Params of modeling
% Dynamic system
h_orb = 400 * 1000;
dt = 1;
t_modeling = 100;
d = dynamics(h_orb, dt);  % Class of dynamics

% Spacecrafts
n_deputy = 3;
t_deploy_deputy = [0, 15, 30];  % in sec
if_deployed_deputy = [false false false];
r_orf_deputy = [[0;0.1;0], [0;0.1;0], [0;0.1;0]];  % in m
v_orf_deputy = [[0;0.1;0], [0;0.1;0], [0;0.1;0]];  % in m/s
i = 1/sqrt(2);
q_irf_deputy = [[-i;i;0;0], [-i;i;0;0], [-i;i;0;0]];
w_orf_deputy = [[0;0;0], [0;0;0], [0;0;0]];

r_orf_chief = [0;0;0];
v_orf_chief = [0;0;0];
q_irf_chief = [1;0;0;0];
w_orf_chief = [0;0;0];
c = spacecraft(d,r_orf_chief,v_orf_chief,q_irf_chief,w_orf_chief);
c.cam_pos = [0;-1.5;0.4];
c.cam_dir = [0;0.7;-0.1];
c.cam_up = [0;0;1];
spacecrafts = [c];  % c variable is no more useful (delete it)

%% Run of display
figure('Color', [1 1 1], 'Position', [200 100 window_size]);
set(gca,'Color','k');
colormap('gray');
camproj('perspective');
axis equal;
cameratoolbar("SetMode","pan");
light('Style', 'local', 'Position', d.R_sun);
set(gca,'visible','off');
gca.XAxis.Visible = 'off';

% Animation
if is_animate
    myVideo = VideoWriter("result");
    myVideo.FrameRate = 60;
    open(myVideo)
end

%% Run of modeling
N = round(t_modeling / dt);

% Docking
names = ["campos", "chief r"];
CameraPosIRF = zeros(N, 3);
ChiefPosIRF = zeros(N, 3);
CameraPosORF = zeros(N, 3);
ChiefPosORF = zeros(N, 3);
modeling_report = table(CameraPosIRF, CameraPosORF, ChiefPosIRF, ChiefPosORF);

for i = 1:N
    % Deploying
    for j = 1:n_deputy
        if and(if_deployed_deputy(j) == false, d.t >= t_deploy_deputy(j))
            if_deployed_deputy(j) = true;
            tmp = spacecraft(d,r_orf_deputy(:,j),v_orf_deputy(:,j), ...
                               q_irf_deputy(:,j),w_orf_deputy(:,j));
            spacecrafts = [spacecrafts, tmp];
        end
    end

    % Time step
    [d, spacecrafts] = d.time_step(spacecrafts);

    clf;
    hold on;

    % Earth show
    [X,Y,Z] = sphere(20);
    fvc = surf2patch(X * d.r_earth, Y * d.r_earth, Z * d.r_earth);
    % patch('Faces', fvc.faces, 'Vertices', fvc.vertices, 'FaceColor', [0.5, 0.5, 0.5])
    
    % Spacecrafts show
    spacecrafts(1).show_chief();
    for j = 2:length(spacecrafts)
        spacecrafts(j).show_deputy();
    end
    
    % Camera update
    cam_pos = spacecrafts(1).get_campos_irf();
    campos(cam_pos);
    camup(spacecrafts(1).get_camup_irf());
    camtarget(cam_pos + spacecrafts(1).get_camdir_irf());
    camva(camera_angle);
    light('Style', 'local', 'Position', d.R_sun);

    axis equal;
    hold off;
    pause(0.001);

    % Docking
    modeling_report.CameraPosIRF(i,:) = cam_pos';
    modeling_report.CameraPosORF(i,:) = d.i2o_r(cam_pos)';
    modeling_report.ChiefPosIRF(i,:) = spacecrafts(1).r_irf';
    modeling_report.ChiefPosORF(i,:) = d.i2o_r(spacecrafts(1).r_irf)';

    % Animation
    if and(i > 1, is_animate)  % Workaround (instead: frame doesn't fit to 900x600)
        frame = getframe(gcf);
        writeVideo(myVideo, frame);
    end
end
if is_animate
    close(myVideo);
end

%% Additional plots
% Plots of trajectories (from table of docking)
% figure('Color', [1 1 1], 'Position', [10 10 900 600]);
% tiledlayout(1,2);
% 
% ax1 = nexttile; 
% plot3(ax1, modeling_report.CameraPosIRF(:,1),modeling_report.CameraPosIRF(:,2),modeling_report.CameraPosIRF(:,3),'--', ...
%       modeling_report.ChiefPosIRF(:,1),modeling_report.ChiefPosIRF(:,2),modeling_report.ChiefPosIRF(:,3));
% legend(["CameraPos", "ChiefPos"]);
% axis equal;
% title(ax1,'IRF trajectories')
% 
% ax2 = nexttile; 
% plot3(ax2, modeling_report.CameraPosORF(:,1),modeling_report.CameraPosORF(:,2),modeling_report.CameraPosORF(:,3),'--', ...
%       modeling_report.ChiefPosORF(:,1),modeling_report.ChiefPosORF(:,2),modeling_report.ChiefPosORF(:,3));
% legend(["CameraPos", "ChiefPos"]);
% title(ax2,'ORF trajectories')

% Plots of constant values (testing)
% figure('Color', [1 1 1], 'Position', [50 50 900 600]);
% hold on;
% plot(modeling_report.CameraPosORF(:,1) - modeling_report.ChiefPosORF(:,1))
% plot(modeling_report.CameraPosORF(:,2) - modeling_report.ChiefPosORF(:,2))
% plot(modeling_report.CameraPosORF(:,3) - modeling_report.ChiefPosORF(:,3))
% hold off;
% legend(["x", "y", "z"]);"""

