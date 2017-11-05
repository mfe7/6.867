function [ smooth_veh_traj ] = find_smooth_veh_trajs( veh_traj, valid_t )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    smooth_veh_traj = [];
    xys = veh_traj{:,{'x','y'}};
    ts = veh_traj{:,{'time'}};
    for ii=1:length(valid_t) % for each [t_start, t_end] window
        ind_start = valid_t(ii,4);
        ind_end = valid_t(ii,5);
%         headings = zeros(ind_end-ind_start+1,1);
        for jj=ind_start:ind_end % for each data pt in a [t_start, t_end]  
            xy = xys(jj,:);
            for kk=ind_start+1:ind_end
                next_xy = xys(kk,:);
                dxdy = next_xy - xy;
                if norm(dxdy) > 0.01
                    heading = atan2(dxdy(2),dxdy(1));
                    r_parallel = normr(dxdy);
%                     headings(jj-ind_start+1) = heading;
                    smooth_veh_traj = [smooth_veh_traj; ts(jj), xy(1), xy(2), heading, r_parallel];
                    break;
                end
            end
        end
%         dh = diff(headings);
%         biggest_dh = max(abs(dh));
%         if (nnz(headings) < length(headings)) && (biggest_dh < 0.2)
%             smooth_veh_traj = [smooth_veh_traj; ts(ind_start:ind_end), xys(ind_start:ind_end,:), headings];
%         end
    end
end

