function [nodes,routes,odpairs,links,vehicle_routes] = getGraphInfo(network_name,adjust,plotting)

if nargin<2
    adjust=false;
end

if nargin<3
    plotting=true;
end

try
    load([network_name])
    if plotting && ~adjust
        for i=1:length(links)
            XYs = [nodes(links(i).connected_nodes(1:2)).XY];
            plot(XYs([1,3]),XYs([2,4]),'k-','LineWidth',2);
        end
        for i=1:length(nodes)
            r = 7;
            fill(nodes(i).XY(1)+r*cos(linspace(0,2*pi,100)),nodes(i).XY(2)+r*sin(linspace(0,2*pi,100)), 'w'); 
            text(nodes(i).XY(1),nodes(i).XY(2),num2str(nodes(i).id),'HorizontalAlignment','center', 'VerticalAlignment','middle','FontSize',18);
        end
    end
    if ~adjust
        return
    else
%         clear('links','routes','odpairs')
%         delete(h1);delete(h2);
    end
catch
end

try

%% Determine layout
if lower(input('Adjust/create layout? [y,N]','s')) == 'y'
    if exist('nodes','var')
        if lower(input('Adjust existing layout? [y,N]','s')) == 'y'
            for i=1:length(nodes)
                x = nodes(i).XY(1);
                y = nodes(i).XY(2);
                ind  = nodes(i).id;
                title(['Ajusting node ', num2str(nodes(i).id),', return to skip'])
                h1 = fill(x+5*cos(linspace(0,2*pi,100)),y+5*sin(linspace(0,2*pi,100)), 'w'); 
                h2 = text(x,y,num2str(ind),'HorizontalAlignment','center', 'VerticalAlignment','middle','FontSize',10);                
                [x,y] = ginput(1);
                if ~isempty(x)
                    delete(h1); delete(h2)
                    nodes(ind).XY = [x,y];
                    fill(x+5*cos(linspace(0,2*pi,100)),y+5*sin(linspace(0,2*pi,100)), 'w'); 
                    text(x,y,num2str(ind),'HorizontalAlignment','center', 'VerticalAlignment','middle','FontSize',10);
                end
            end
        else
            for i=1:length(nodes)
                x = nodes(i).XY(1);
                y = nodes(i).XY(2);
                ind  = nodes(i).id;
                fill(x+5*cos(linspace(0,2*pi,100)),y+5*sin(linspace(0,2*pi,100)), 'w'); 
                text(x,y,num2str(ind),'HorizontalAlignment','center', 'VerticalAlignment','middle','FontSize',10);
            end
        end        
    else
        ind = 0;
    end
    
    [x,y] = ginput(1);
    while ~isempty(x)
        ind = ind+1;
        nodes(ind).id = ind;
        nodes(ind).XY = [x,y];
        fill(x+5*cos(linspace(0,2*pi,100)),y+5*sin(linspace(0,2*pi,100)), 'w'); 
        text(x,y,num2str(ind),'HorizontalAlignment','center', 'VerticalAlignment','middle','FontSize',10);
        [x,y] = ginput(1);
    end
else
    for i=1:length(nodes)
        fill(nodes(i).XY(1)+5*cos(linspace(0,2*pi,100)),nodes(i).XY(2)+5*sin(linspace(0,2*pi,100)), 'w'); 
        text(nodes(i).XY(1),nodes(i).XY(2),num2str(nodes(i).id),'HorizontalAlignment','center', 'VerticalAlignment','middle','FontSize',8);
    end
end
save(['../common_files/',network_name],'nodes','-append')

%% Graph structure:
if lower(input('Adjust node connectivity? [y,N]','s')) == 'y'
    num_nodes = length(nodes);
    for i=1:num_nodes
        neighbors = [];
        if isfield(nodes(i),'neighbors')
            neighbors = nodes(i).neighbors;
            for n=neighbors
                XYs = [nodes([i,n]).XY];
                h1{n} = plot(XYs([1,3]),XYs([2,4]),'k-','LineWidth',2);
            end
        end
        display(['Current neighbors = ', num2str(nodes(i).neighbors)])
        neighbor_input = input(['Neighbor of ',num2str(i),': ']);
        while ~isempty(neighbor_input)
            if ismember(neighbor_input,neighbors)
                neighbors(neighbors == neighbor_input) = [];
                delete(h1{neighbor_input})
            else
                neighbors = [neighbors , neighbor_input];
                XYs = [nodes([i,neighbor_input]).XY];
                h1{neighbor_input} = plot(XYs([1,3]),XYs([2,4]),'k-','LineWidth',2);
            end
            display(['Current neighbors = ', num2str(nodes(i).neighbors)])
            neighbor_input = input(['Neighbor of ',num2str(i),': ']);            
        end 
        nodes(i).neighbors = neighbors;
        cellfun(@delete,h1)
    end
end

%Link check
for i=1:length(nodes)
    for j=nodes(i).neighbors
        if isempty(find(ismember(nodes(j).neighbors,i),1))
            warning =  sprintf('Warning: Node %i is a neighbor of %i, but not the other way around. Add it? [y,N]',j,i);
            if lower(input(warning,'s')) == 'y'
                nodes(j).neighbors(end+1) = i;
            end
        end
    end
end


%% Enumerate links:
if lower(input('Rebuild links? [y,N]','s')) == 'y'
    clear links
    id_L=0;
    for i=1:length(nodes)
        for j=1:length(nodes(i).neighbors)
            id_L = id_L+1;
            links(id_L) =  struct('id',id_L,'connected_nodes',[nodes(i).id, nodes(i).neighbors(j)]);
        end
    end
end    
for i=1:length(links)
    XYs = [nodes(links(i).connected_nodes(1:2)).XY];
    plot(XYs([1,3]),XYs([2,4]),'k-','LineWidth',2);
end


if lower(input('Set link velocities? [y,N]','s')) == 'y'
    for i=1:length(links)
        XYs = [nodes(links(i).connected_nodes(1:2)).XY];
        hl = plot(XYs([1,3]),XYs([2,4]),'g-','LineWidth',4);
        input_vel = input(['Velocity of link between nodes ',num2str(links(i).connected_nodes),', (previous is ', num2str(links(i).vel), ', press enter to keep): ']);
        if ~isempty(input_vel)
            links(i).vel = input_vel;
        end
        delete(hl)
    end
end

%Find opposite link
opposite_links_cell = cellfun(@(x) find(~any(fliplr(vertcat(links.connected_nodes)) - repmat(x,length(links),1),2)) , {links.connected_nodes},'UniformOutput',false);
[links.opposite_link] = opposite_links_cell{:};

%Check symetry in link speeds
for i=1:length(links)
    if links(i).vel~=links(links(i).opposite_link).vel
        XYs = [nodes(links(i).connected_nodes(1:2)).XY];
        hl = plot(XYs([1,3]),XYs([2,4]),'g-','LineWidth',4);
        input_vel = (['Velocity of link between nodes ',num2str(links(i).connected_nodes),' is ', num2str(links(i).vel), ' but its opposite is ', links(links(i).opposite_link).vel,', input common vel or press enter to keep as is: ']);
        if ~isempty(input_vel)
            links(i).vel = input_vel;
            links(links(i).opposite_link).vel = input_vel;
        end
        delete(hl)
    end
end

[nodes.vehicle_neighbors] = nodes.neighbors;
% Create vehicle neighbors
for l=find([links.vel]==0);
    node = links(l).connected_nodes(1);
    neighbor = links(l).connected_nodes(2);
    nodes(node).vehicle_neighbors(nodes(node).vehicle_neighbors==neighbor) = [];
end

%% Define origin nodes:
if lower(input('Adjust origin/destinations? [y,N]','s')) == 'y'
    if lower(input('All nodes are origins and destinations? [y,N]','s')) == 'y'
        for i=1:length(nodes)
            nodes(i).origin = true;
            nodes(i).destinations = true;
        end
    else
        display('List origin nodes one by one to add (default) or remove them, leave blank when finished:')
        origin_input = input('Origin node: ');
        origin_nodes = [];
        while ~isempty(origin_input)
            if ismember(origin_input,origin_nodes)
                origin_nodes(origin_nodes == origin_input) = [];
            else
                origin_nodes = [origin_nodes , origin_input];
            end
            origin_input =  input('Origin node: ');
        end
        for i=1:length(nodes)
            if ismember(i,origin_nodes)
                nodes(i).origin = true;
            else
                nodes(i).origin = false;
            end
        end

        %Define destination nodes:
        if (lower(input('Destination nodes same as origin nodes? (y/N)','s')) == 'y')
            for i=1:length(nodes)
                    nodes(i).destination = nodes(i).origin;
            end
        else
            display('List destination nodes one by one to add (default) or remove them, leave blank when finished:')
            destination_input = input('Destination node: ');
            destination_nodes = [];
            while ~isempty(destination_input)
                if ismember(destination_input,destination_nodes)
                    destination_nodes(destination_nodes == destination_input) = [];
                else
                    destination_nroute_finderodes = [destination_nodes , destination_input];
                end
                destination_input =  input('Destination node: ');
            end
            for i=1:length(nodes)
                if ismember(i,destination_nodes)
                    nodes(i).destination = true;
                else
                    nodes(i).destination = false;
                end
            end
        end
    end
end

%% Get line end points for each link
for i=1:length(links)
    x1 = nodes(links(i).connected_nodes(1)).XY(1);
    y1 = nodes(links(i).connected_nodes(1)).XY(2);
    x2 = nodes(links(i).connected_nodes(2)).XY(1);
    y2 = nodes(links(i).connected_nodes(2)).XY(2);
    links(i).points = [linspace(x1,x2,100);linspace(y1,y2,100)];
    links(i).link_length = sqrt(sum(diff(links(i).points(:,[1,end]),1,2).^2));
    links(i).heading = atan2(diff(links(i).points(2,[1,end])),diff(links(i).points(1,[1,end])));
end

%% Enumerate routes:
if lower(input('Rebuild routes? [y,N]','s')) == 'y'
clear routes vehicle_routes
origin_nodes = find([nodes.origin] == 1);
destination_nodes = find([nodes.destination] == 1);
id_OD = 0;
for origin_node = origin_nodes
    for destination_node = setdiff(destination_nodes,origin_node)
        id_OD = id_OD + 1;
        new_routes = route_finder(nodes, origin_node, destination_node);
        [min_length,min_time] = get_best_routes(new_routes,links);
        if isinf(min_time.time)
            new_routes = route_finder_time(nodes, origin_node, destination_node);
            [~,min_time] = get_best_routes(new_routes,links);
        end                         
        routes(id_OD) = struct('id',id_OD,'connected_nodes',min_length.route,'connected_links',min_length.cl,'links_incidence',min_length.li,'origin',origin_node,'destination',destination_node,'length',min_length.length);
        vehicle_routes(id_OD) = struct('id',id_OD,'connected_nodes',min_time.route,'connected_links',min_time.cl,'links_incidence',min_time.li,'origin',origin_node,'destination',destination_node,'length',min_time.length,'time',min_time.time);
    end
end
end

%%
save(['../common_files/',network_name],'nodes','links','routes','odpairs','vehicle_routes')

catch e
    e.stack.line
    rethrow(e)
end

end

function [min_length,min_time] = get_best_routes(new_routes,links)
    route_length = zeros(length(new_routes),1);
    route_time = zeros(length(new_routes),1);
    li = cell(length(new_routes),1);
    for i = 1:length(new_routes)
        num_links = length(new_routes{i})-1;
        links_incidence = zeros(size(links))';
        connected_links = zeros(1,num_links);
        for j = 1:num_links
            link_ind = ismember(vertcat(links.connected_nodes),new_routes{i}(j:j+1),'rows');
            connected_links(j) = find(link_ind);
            links_incidence = links_incidence + link_ind;
        end
        li{i} = links_incidence;
        cl{i} = connected_links;
        route_length(i) = sum([links(find(links_incidence)).link_length]); %#ok<FNDSB>
        route_time(i) = sum([links(find(links_incidence)).link_length]./[links(find(links_incidence)).vel]); %#ok<FNDSB>
    end
    [~,shortest_route_ind] = min(route_length);
    [~,shortest_time_ind] = min(route_time);
    
    min_length.route = new_routes(shortest_route_ind);
    min_length.length = route_length(shortest_route_ind);
    min_length.li = li{shortest_route_ind};
    min_length.cl = cl{shortest_route_ind};
    
    min_time.route = new_routes(shortest_time_ind);
    min_time.length = route_length(shortest_time_ind);
    min_time.time = route_time(shortest_time_ind);
    min_time.li = li{shortest_time_ind};
    min_time.cl = cl{shortest_route_ind};
end