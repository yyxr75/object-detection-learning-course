% plot_range_azimuth_3D.m
%
% Function to plot range and azimuth heat map

%input
%   frame_id:           当前帧数
%   range_resolution: range resolution to calculate axis to plot
%   radar_data_pre_3dfft: input 3D matrix, rangeFFT x DopplerFFT x virtualArray
%   TDM_MIMO_numTX: number of TXs used for processing
%   numRxAnt: : number of RXs used for processing
%   antenna_D: array ID，矩阵形式
%   LOG: 1:plot non-linear scale, ^0.4 by default
%   PLOT_ON: 1 = plot on; 0 = plot off
%   minRangeBinKeep: start range index to keep
%   rightRangeBinDiscard: number of right most range bins to discard
%   angles_DOA_az: 水平角度估计范围，要不就[-70,0.5,70]
%   angles_DOA_ele: 垂直角度估计范围，要不就[-20,0.5,20]
%   detection_obj: CFAR得到的目标点
%   DYNAMIC: 采用何种方法计算动态目标角度 1-CBF
%   STATIC: 静止目标的DOA估计方法。1-CBF

%output
%   mag_data_static: zero Doppler range/azimuth heatmap
%   mag_data_dynamic: non-zero Doppler range/azimuth heatmap
%   y_axis: y axis used for visualization（这个可能不返回了，出图的部分直接集成在函数里了
%   x_axis: x axis used for visualization（这个可能不返回了，出图的部分直接集成在函数里了


function  [mag_data_static mag_data_dynamic y_axis x_axis] = plot_range_azimuth_3D(frame_id, range_resolution, radar_data_pre_3dfft,TDM_MIMO_numTX,numRxAnt,...
    antenna_D, LOG, PLOT_ON, minRangeBinKeep, rightRangeBinDiscard, angles_DOA_az, angles_DOA_ele, detection_obj, DYNAMIC, STATIC)
    %% 计算Range Angle 热力图
    
    % 默认参数赋值
    if ~exist('DYNAMIC','var')
        DYNAMIC = 1;
    end
    if ~exist('STATIC','var')
        STATIC = 1;
    end
    
    % 参数初始化
    dopplerFFTSize = size(radar_data_pre_3dfft,2);
    rangeFFTSize = size(radar_data_pre_3dfft,1);
    D = antenna_D;
    
    % 天线坐标，用于方位角和垂直角估计（天线坐标见级联雷达文档）    
    D = D + 1;
    apertureLen_azim = max(D(:,1));
    apertureLen_elev = max(D(:,2));
    
    % 取出速度为0的bin，把动态和静态拆开
    % 取不同模式的RangeAngle
    radar_data_dynamic = radar_data_pre_3dfft;                          % 运动部分，先取全部的doppler bin，一会儿通过运动的目标点进行筛选
    radar_data_Static = radar_data_pre_3dfft(:,dopplerFFTSize/2+1,:);   % 静止部分的RangeAngle，在速度为0处选取
    
    % 去掉坐标重复的天线，并根据下标重新调整信号矩阵的顺序
    antenna_uq = []; % 去掉坐标重复的天线
    ind_uq = [];     % 去掉坐标重复的天线后，相对于原本信号的下标
    for i_line = 1:apertureLen_elev
        % 从第一行虚拟天线开始，每一行取出对应该天线的数据，去掉重复项，再组合到一起
        ind = find(D(:,2) == i_line);
        D_sel = D(ind,1);
        [~, indU] = unique(D_sel);
        % 为了避免水平方向的加权，应当去掉横坐标相同的阵元
%         for idx = 1:(i_line-1)
%             i_D_sel = 1;
%             while i_D_sel <= size(indU, 1)               % 逐一比较当前行阵元的横坐标和先前的横坐标
%                 ant_ind = find(antenna_uq(:, 2) == idx); % 从最后一行开始，遍历寻找横坐标相同的阵元
%                 if size(ant_ind, 1) < 1
%                     break;
%                 end
%                 ant = antenna_uq(ant_ind, 1);            % 取横坐标
%                 d_i = D_sel(indU(i_D_sel), 1);
%                 find_d_id = find(ant == d_i);           % 寻找相同横坐标的阵元
%                 if size(ant, 1) > 1                     % 若先前的阵元所在行还有其他阵元，则去掉先前的阵元
%                     antenna_uq(ant_ind(find_d_id), :) = [];
%                     ind_uq(ant_ind(find_d_id), :) = [];
%                 else                                    % 若先前的阵元所在行没有其他阵元了，则去掉当前行的阵元
%                      indU(i_D_sel) = [];
%                      i_D_sel = i_D_sel - 1;
%                 end
%                 
%                 i_D_sel = i_D_sel + 1;
%             end
            
%         end
        antenna_uq = [antenna_uq; D(ind(indU),:)];
        ind_uq = [ind_uq; ind(indU)];
    end
    antenna_uq = antenna_uq - 1;
    % 调整信号矩阵的顺序
    radar_data_Static = squeeze(radar_data_Static);
    radar_data_Static = radar_data_Static(:, ind_uq);% 
    
    % 基于天线坐标矩阵生成每个角度(theta, phi)的波束向量，并形成矩阵
    thetas = angles_DOA_az(1):angles_DOA_az(2):angles_DOA_az(3);
    phis = angles_DOA_ele(1):angles_DOA_ele(2):angles_DOA_ele(3);
    steer_vectors = zeros(size(thetas,2), size(phis,2), size(antenna_uq,1));    % 波束向量初值，size = [theta, phi, antenna]
    for i_phi = 1:size(phis, 2) % 垂直角度
        phi = phis(1, i_phi);
        for i_theta = 1:size(thetas, 2) % 水平角度
            theta = thetas(1, i_theta);
            steer = array_response_vector_3d(antenna_uq, theta, phi);
            steer_vectors(i_theta, i_phi, :) = steer;
        end
    end
    
    % 静态部分：
    % 在每个距离bin，扫描每个角度bin，计算波束形成
    if STATIC == 1
        fprintf('CBF-3D\n');
        tic
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1), size(thetas,2), size(phis,2));
        % 逐一距离计算角度
        for i = 1:rangeFFTSize
%         for i = 105
            radar_data_pre_cbf = squeeze(radar_data_Static(i,:)).';
            X = radar_data_pre_cbf * radar_data_pre_cbf';
%           % 逐角度计算CBF
                for i_phi = 1:size(phis, 2) % 垂直角度
                    for i_theta = 1:size(thetas, 2) % 水平角度
                        steer = squeeze(steer_vectors(i_theta, i_phi, :));% 波束向量
                        P = steer' * X * steer;
                        radar_data_angle_range_Static(i, i_theta, i_phi) = P;
                    end
                end
        end
        toc
        n_phis_size = size(radar_data_angle_range_Static,3);
        n_theta_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
    end
    
    % 动态部分：暂缓
    if DYNAMIC == 1
        fprintf('DYNAMIC == 1');
    end
    
    % 取部分有效的Range下标
    indices_1D = (minRangeBinKeep:n_range_fft_size-rightRangeBinDiscard);
    max_range = (n_range_fft_size-1)*range_resolution;
    max_range = max_range/2;
    d = 1;
    
    %%% 单独显示某一距离bin上的az-el热力图
    [Th, Ph] = meshgrid(thetas, phis);
    figure(5);
    mesh(Th, Ph, abs(squeeze(radar_data_angle_range_Static(105,:,:))'));
    
    % 把3D数据在垂直方向上累加，做成平面的
%     radar_data_angle_range_Static_az = squeeze(max(radar_data_angle_range_Static, [], 3));
    radar_data_angle_range_Static_az = squeeze(radar_data_angle_range_Static(:, :, floor(n_phis_size / 2 + 1)));
    mag_data_static_RA = squeeze(abs(radar_data_angle_range_Static_az(indices_1D+1, :)));
    % 每个距离逐一对角度进行归一化
    % 静态（如果不用CBF，先注释掉）
    for idx_rge = 1:size(mag_data_static_RA, 1)
       temp = mag_data_static_RA(idx_rge, :);
       max_temp = max(temp);
       temp = temp./max_temp;
       temp = temp.*(20*(max_temp^0.6));
%        temp = temp.*(20*log10(max_temp));
       mag_data_static_RA(idx_rge, :) = temp;
    end
    
    % TODO: 确定坐标轴
    % 这里需要改成角度上均衡的
    theta_ = thetas;
    phi_ = phis;
    axis_range = indices_1D*range_resolution;   % range维度
    axis_angle = theta_;                        % angle维度
    [R_mat, sine_theta_mat] = meshgrid(axis_range, sin(axis_angle*pi/180));
    [~, cos_theta_mat] = meshgrid(axis_range, cos(axis_angle*pi/180));
    x_axis = R_mat.*cos_theta_mat;
    y_axis = R_mat.*sine_theta_mat;
    
    % 显示range angle图(叠在一起的）
    [X,Y] = meshgrid(axis_range,axis_angle);    
    figure(4);
    subplot(121);
    mesh(Y',X',(mag_data_static_RA).^0.4);
    xlabel('angle-degree');        
    ylabel('range-meter');        
    title('Range-Angle Map');
    view(2);
    subplot(122);
    surf(y_axis, x_axis, ((mag_data_static_RA).^0.4)','EdgeColor','none');%% 显示
    view(2);
    xlabel('meters-x');
    ylabel('meters-y');
    title('range/azimuth heat map static objects');
    
    
    %%% 以下是粘过来的原程序，待修改
    

    % TODO: 最终结果提取峰值？可视化？这个还得做
    
    %%% 以下是粘过来的原程序，待修改
    
