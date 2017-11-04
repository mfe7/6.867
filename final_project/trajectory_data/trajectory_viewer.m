clear; close all; clc;

addpath(genpath('.'))

%% Parameters
load('april')
length_threshold = 1; %min length of path
use_google_map = 1;
network_name = 'stata_feb_utm';
network_name = 'stata_kendall_green';
static = '';

%% Load map

figure('name','data'); 
% if use_google_map
%     imshow(flipud(street_map.image),street_map.ref)
% else
%     imshow(flipud(grid_map.image),grid_map.ref)
% end
rectangle('Position',[-1 -3 2 3],'EdgeColor','blue');
rectangle('Position',[-1 0 2 10],'LineStyle','--','EdgeColor','red');
set(gca,'ydir','normal')
hold on;
pause(1e-9);
% axis off

%% Load graph
% [nodes,routes,odpairs,links] = getGraphInfo(network_name,0);

%% Find all trajectory files and import them as a table
clusters_folder = ['clusters_',network_name,static];
files = dir(clusters_folder);
for i=3:length(files)
    files(i).name
    load(files(i).name)
    if i==3
        clusters_all = clusters2;
    elseif ~isempty(clusters2)
        clusters_all = [clusters_all, clusters2];
    end
end
clusters = clusters_all;
clear('clusters_all');



%% Plot a single ped trajectory + golf cart

%% Plot trajectores
ind = 0;
if use_google_map
    for i=1:length(clusters)
        if mod(i,0)==0 || i>0.7*length(clusters) %clusters(i).velocity<5
            h{1}(i) = plot(clusters(i).local_x, clusters2(i).local_y,'color',clusters2(i).color,'LineWidth',2);
            
%             h{1}(i) = plot(clusters(i).easting,clusters(i).northing,'color',clusters(i).color,'LineWidth',2);
%             h{2}(i) = plot(clusters(i).easting(end),clusters(i).northing(end),'o','color',clusters(i).color,'LineWidth',2);
            drawnow;
            pause(1);
        end
    end
else
%     for i=1:length(clusters)
%         if norm([clusters(i).x(end),clusters(i).y(end)]-[clusters(i).x(1),clusters(i).y(1)]) >= length_threshold
%             h{1}(i) = plot(clusters(i).x,clusters(i).y,'color',clusters(i).color,'LineWidth',2);
%             h{2}(i) = plot(clusters(i).x(end),clusters(i).y(end),'o','color',clusters(i).color,'LineWidth',2);
%         end
%     end
end
drawnow;
display("done");

% Load graph
% [nodes,routes,odpairs,links] = getGraphInfo(network_name,0);

%% Save figure
% fig_name = 'network_paths_with_graph';
% fig2Pdf([fig_name],600)


