function [ smooth_veh_traj, smooth_valid_t ] = find_smooth_veh_traj( veh_traj, valid_t )
% find_smooth_veh_traj: Turn a jumpy vehicle trajectory into a smooth one 
%   valid_t has the upper/lower indices of windows where the vehicle
%   trajectory doesn't jump in position/time.
%   Loop through the valid_t windows and compute current vehicle heading.
%   Must also remove defects like when the vehicle position moves backward.
%
%   In addition to just populating a new data structure, the hard part of
%   this function is computing the vehicle's heading, and this is really
%   important to accurately transform the pedestrian trajectory later on.

    should_plot = 0;

    % 1) Extract (x,y,timestamp) from raw vehicle trajectory
    smooth_veh_traj = [];
    smooth_valid_t = [];
    xys = veh_traj{:,{'x','y'}};
    ts = veh_traj{:,{'time'}};
    latlon = veh_traj{:,{'easting','northing'}};
    
    % 2) Treat each valid_t window independently, and add the relevant 
    %       fields to smooth_veh_traj:
    % smooth_veh_traj =
    %   [timestamp, global_x, global_y, heading (rad), r_par, r_orthog, easting, northing]
    %       where r_par, r_orthog is the unit vector pointing out of the
    %       vehicle's front
    %       (one row per timestamp)
    
    % Iterate through every valid_t window [t_start, t_end]
    for ii=1:length(valid_t)
%     tmp = 5;
%     for ii=tmp:tmp
        ind_start = valid_t(ii,4);
        ind_end = valid_t(ii,5);
        
        % Compute distance traveled during segment. Skip that segment
        % if vehicle didn't move (bc no way to compute heading)
        xy = xys(ind_start:ind_end,:);
        dxdy = diff(xy);
        dist_traveled = sum(sqrt(sum(dxdy.^2,2)));
        if dist_traveled == 0
            display(strcat('Vehicle didn''t move during segment #',num2str(ii)));
            continue;
        end
        
        % Iterate through each data pt in a [t_start, t_end] window
        for jj=ind_start:ind_end
            xy = xys(jj,:);
            for kk=jj+1:ind_end
                next_xy = xys(kk,:);
                dxdy = next_xy - xy;
                if norm(dxdy) > 0.01
                    heading = atan2(dxdy(2),dxdy(1));
                    valid_heading = 1;
                    if jj > ind_start
                        prev_heading = smooth_veh_traj(end, 4);
                        if abs(mod(heading - prev_heading, 2*pi)) > 0.3
                            valid_heading = 0;
                        end
                    end
                    if valid_heading
                        r_parallel = normr(dxdy);
                        smooth_veh_traj = [smooth_veh_traj; ts(jj), xy(1), xy(2), heading, r_parallel, latlon(jj,1), latlon(jj,2)];
                        break;
                    end
                end
            end
            
            if should_plot == 1
                hold on;
                subplot(1,2,1);
                plot(xy(1), xy(2),'x');
                quiver(xy(1),xy(2),r_parallel(1),r_parallel(2),0);
                title(strcat('jj=',num2str(jj)));
                hold on;
                subplot(1,2,2);
                plot(ts(jj),heading,'x');
                pause(0.05);

            end
        end
        smooth_valid_t = [smooth_valid_t; valid_t(ii,:)];
%         dh = diff(headings);
%         biggest_dh = max(abs(dh));
%         if (nnz(headings) < length(headings)) && (biggest_dh < 0.2)
%             smooth_veh_traj = [smooth_veh_traj; ts(ind_start:ind_end), xys(ind_start:ind_end,:), headings];
%         end
    end
end

