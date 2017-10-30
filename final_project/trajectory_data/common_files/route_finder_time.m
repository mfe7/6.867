function routes = route_finder_time(nodes, route_so_far, destination)

potential_next_neighbors = nodes(route_so_far(end)).vehicle_neighbors;

if ismember(destination,potential_next_neighbors)
    routes = {[route_so_far,destination]};
    return
end

routes = {};
for i=1:length(potential_next_neighbors)
    
    %Check if neighbor has already been visited
    if ismember(potential_next_neighbors(i),route_so_far)     
        continue; %Skip this neighbor
    end
    
    %Check if neighbor of neighbor has been visited indicating a shorter path
    neighbors_of_potential_next_neighbor = nodes(potential_next_neighbors(i)).vehicle_neighbors;
    if length(find(ismember(neighbors_of_potential_next_neighbor,route_so_far))) > 1 % If more than one neighbor is on the route_so_far then there is a shorter path to here
        continue; %Skip this neighbor
    end
    
    %Otherwise neighbor is valid and route search continues
    new_routes = route_finder_time(nodes, [route_so_far, potential_next_neighbors(i)] , destination);
    for j = 1:length(new_routes)
        routes{end+1} = new_routes{j};
    end
end
            
    
   