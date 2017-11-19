function [ smooth_veh_traj ] = find_smooth_veh_traj( veh_traj, valid_t )
% find_smooth_veh_traj: Turn a jumpy vehicle trajectory into a smooth one 
%   valid_t has the upper/lower indices of windows where the vehicle
%   trajectory doesn't jump in position/time.
%   Loop through the valid_t windows and compute current vehicle heading.
%   Must also remove defects like when the vehicle position moves backward.
%
%   In addition to just populating a new data structure, the hard part of
%   this function is computing the vehicle's heading, and this is really
%   important to accurately transform the pedestrian trajectory later on.


    % 1) Extract (x,y,timestamp) from raw vehicle trajectory
    smooth_veh_traj = [];
    xys = veh_traj{:,{'x','y'}};
    ts = veh_traj{:,{'time'}};
    
    % 2) Treat each valid_t window independently, and add the relevant 
    %       fields to smooth_veh_traj:
    % smooth_veh_traj =
    %   [timestamp, global_x, global_y, heading (rad), r_par, r_orthog]
    %       where r_par, r_orthog is the unit vector pointing out of the
    %       vehicle's front
    %       (one row per timestamp)
    
    % Iterate through every valid_t window [t_start, t_end]
    for ii=1:length(valid_t)
% TODO: Fix case of no motion through whole window...
%     for ii=2:2
        ind_start = valid_t(ii,4);
        ind_end = valid_t(ii,5);
        
        % Iterate through each data pt in a [t_start, t_end] window
        
%         figure;
        for jj=ind_start:ind_end
            xy = xys(jj,:);
            for kk=ind_start+1:ind_end
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
                        smooth_veh_traj = [smooth_veh_traj; ts(jj), xy(1), xy(2), heading, r_parallel];
                        break;
                    end
                end
            end
            
%             hold on;
%             subplot(1,2,1);
%             plot(xy(1), xy(2),'x');
%             title(strcat('jj=',num2str(jj)));
%             hold on;
%             subplot(1,2,2);
%             plot(ts(jj),heading,'x');
%             pause(0.05);
        end
%         dh = diff(headings);
%         biggest_dh = max(abs(dh));
%         if (nnz(headings) < length(headings)) && (biggest_dh < 0.2)
%             smooth_veh_traj = [smooth_veh_traj; ts(ind_start:ind_end), xys(ind_start:ind_end,:), headings];
%         end
    end
end

