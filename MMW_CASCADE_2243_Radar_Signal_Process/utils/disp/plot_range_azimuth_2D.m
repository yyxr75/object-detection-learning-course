%  Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
%
%
%   Redistribution and use in source and binary forms, with or without
%   modification, are permitted provided that the following conditions
%   are met:
%
%     Redistributions of source code must retain the above copyright
%     notice, this list of conditions and the following disclaimer.
%
%     Redistributions in binary form must reproduce the above copyright
%     notice, this list of conditions and the following disclaimer in the
%     documentation and/or other materials provided with the
%     distribution.
%
%     Neither the name of Texas Instruments Incorporated nor the names of
%     its contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
%   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
%   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
%   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
%   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
%   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%

% plot_range_azimuth_2D.m
%
% Function to plot range and azimuth heat map

%input
%   range_resolution: range resolution to calculate axis to plot
%   radar_data_pre_3dfft: input 3D matrix, rangeFFT x DopplerFFT x virtualArray
%   TDM_MIMO_numTX: number of TXs used for processing
%   numRxAnt: : number of RXs used for processing
%   antenna_azimuthonly: azimuth array ID
%   LOG: 1:plot non-linear scale, ^0.4 by default
%   STATIC_ONLY: 1 = plot heatmap for zero-Doppler; 0 = plot heatmap for nonzero-Doppler
%   PLOT_ON: 1 = plot on; 0 = plot off
%   minRangeBinKeep: start range index to keep
%   rightRangeBinDiscard: number of right most range bins to discard
%   detection_obj: CFAR得到的目标点
%   DYNAMIC: 采用何种方法计算动态目标角度 0-FFT，1-MUSIC
%   STATIC: 静止目标的DOA估计方法。0-FFT，1-CBF，2-Capon，3-Compressed Sensing OMP，4-Compressed Sensing L1 Optimization

%output
%   mag_data_static: zero Doppler range/azimuth heatmap
%   mag_data_dynamic: non-zero Doppler range/azimuth heatmap
%   y_axis: y axis used for visualization
%   x_axis: x axis used for visualization



function  [mag_data_static mag_data_dynamic y_axis x_axis] = plot_range_azimuth_2D(frame_id, range_resolution, radar_data_pre_3dfft,TDM_MIMO_numTX,numRxAnt,...
    antenna_azimuthonly, LOG, STATIC_ONLY, PLOT_ON, minRangeBinKeep, rightRangeBinDiscard,detection_obj, DYNAMIC, STATIC)
    %% 计算Range Angle 热力图
    
    % 默认参数赋值
    if ~exist('DYNAMIC','var')
        DYNAMIC = 0;
    end
    if ~exist('STATIC','var')
        STATIC = 0;
    end

    % 参数初始化
    dopplerFFTSize = size(radar_data_pre_3dfft,2);
    rangeFFTSize = size(radar_data_pre_3dfft,1);
    angleFFTSize = 256;
    % ratio used to decide engergy threshold used to pick non-zero Doppler bins
    ratio = 0.5;
    DopplerCorrection = 0;

    % 是否进行相位校正
    if DopplerCorrection == 1
        % add Doppler correction before generating the heatmap
        %% doppler相位校正（TDM分时补偿，保证角度估计的正确性），但是在之前完成dopplerFFT之后已经进行了一次相位校正，所以在这里不再重复校正
        radar_data_pre_3dfft_DopCor= [];
        for dopplerInd = 1: dopplerFFTSize
            deltaPhi = 2*pi*(dopplerInd-1-dopplerFFTSize/2)/( TDM_MIMO_numTX*dopplerFFTSize);
            sig_bin_org =squeeze(radar_data_pre_3dfft(:,dopplerInd,:));
            for i_TX = 1:TDM_MIMO_numTX
                RX_ID = (i_TX-1)*numRxAnt+1 : i_TX*numRxAnt;
                corVec = repmat(exp(-1j*(i_TX-1)*deltaPhi), rangeFFTSize, numRxAnt);
                radar_data_pre_3dfft_DopCor(:,dopplerInd, RX_ID)= sig_bin_org(:,RX_ID ).* corVec;
            end
        end

        radar_data_pre_3dfft = radar_data_pre_3dfft_DopCor;
    end
    
    % 取处于同一水平线上的天线数据
    radar_data_pre_3dfft = radar_data_pre_3dfft(:,:,antenna_azimuthonly);
    
    % 计算Hanning窗函数
    AngLen = size(radar_data_pre_3dfft, 3); % 窗的尺寸
    Angle_Win = hanning(AngLen);            % Hanning窗
    Angle_Win = Angle_Win(1: (AngLen / 2)); % 取前一半
    AngleWinLen_vel = length(Angle_Win);    % 前一半的长度
    AngleWindowCoeffVec_vel = ones(AngLen, 1);              % Hanning窗向量
    AngleWindowCoeffVec_vel(1:AngleWinLen_vel) = Angle_Win; % 前一半赋值
    AngleWindowCoeffVec_vel(AngLen-AngleWinLen_vel+1:AngLen) = AngleWindowCoeffVec_vel(AngleWinLen_vel:-1:1);   % 后一半赋值
    Angle_Win = zeros(1, 1, AngLen);
    Angle_Win(1, 1, :) = AngleWindowCoeffVec_vel;  % 最终得到的窗函数
    
    % 取不同模式的RangeAngle
    radar_data_dynamic = radar_data_pre_3dfft;                          % 运动部分，先取全部的doppler bin，一会儿通过运动的目标点进行筛选
    radar_data_Static = radar_data_pre_3dfft(:,dopplerFFTSize/2+1,:);   % 静止部分的RangeAngle，在速度为0处取模长
    
    %%% 保存radar_data_Static部分数据，用后即删
%     save_mat = ['./save_data/g6-test/', num2str(frame_id), '.mat'];
%     save(save_mat, 'radar_data_Static');
    
    if ~STATIC
        %% 不使用其他方法，只采用FFT
        %%% 静态部分，依然采用FFT %%%
        % 为数据加窗
%         radar_data_pre_3dfft_hanning = bsxfun(@times, radar_data_Static, Angle_Win);
%         radar_data_Static = radar_data_pre_3dfft_hanning;
        % 角度FFT
        radar_data_angle_range = fft(radar_data_Static, angleFFTSize, 3);
        n_angle_fft_size = size(radar_data_angle_range,3);
        n_range_fft_size = size(radar_data_angle_range,1);
        % 取模长
        radar_data_angle_range_Static = squeeze(abs(radar_data_angle_range(:,1,:)));% 静止部分的RangeAngle
        % 角度有正负，取shiftFFT
        radar_data_angle_range_Static = fftshift(radar_data_angle_range_Static,2);
    elseif STATIC == 1
        %% 采用CBF计算静态部分角度
        fprintf('CBF\n');
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1),angleFFTSize);
        
        % 为数据加窗
%         radar_data_pre_3dfft_hanning = bsxfun(@times, radar_data_Static, Angle_Win);
%         radar_data_Static = radar_data_pre_3dfft_hanning;
        % 每一个rangebin，逐一进行CBF计算
        for i=1:size(radar_data_Static,1)
            radar_data_pre_cbf = squeeze(radar_data_Static(i,:));
            radar_data_angle_range_Static(i,:) = CBFAlg(radar_data_pre_cbf', angleFFTSize);
        end
        n_angle_fft_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
    
    elseif STATIC == 2
        %% 采用Capon计算静态部分角度
        fprintf('capon\n');
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1),angleFFTSize);
        
        % 每一个rangebin，逐一进行capon计算
        for i=1:size(radar_data_Static,1)
            radar_data_pre_capon = squeeze(radar_data_Static(i,:));
            radar_data_angle_range_Static(i,:) = CaponAlg(radar_data_pre_capon', angleFFTSize);
        end
        n_angle_fft_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
        
    elseif STATIC == 3
        %% 采用 Compressed Sensing OMP 计算静态部分角度
        fprintf('Compressed Sensing OMP\n');
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1),angleFFTSize);
        
        % 每一个rangebin，逐一进行Compressed Sensing OMP计算
        for i=1:size(radar_data_Static,1)
            fprintf('%d ',i);
            radar_data_pre_omp = squeeze(radar_data_Static(i,:));
            radar_data_angle_range_Static(i,:) = CS_OmpAlg(radar_data_pre_omp', angleFFTSize);
        end
        fprintf('\n');
        n_angle_fft_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
    
    elseif STATIC == 4
        %% 采用Compressed Sensing L1范数约束优化求解
        fprintf('Compressed Sensing L1\n');
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1),angleFFTSize);
        
        % 每一个rangebin，逐一进行Compressed Sensing L1计算
        for i=1:size(radar_data_Static,1)
            fprintf('%d ',i);
            radar_data_pre_csl1 = squeeze(radar_data_Static(i,:));
            radar_data_angle_range_Static(i,:) = CS_L1Alg(radar_data_pre_csl1', angleFFTSize);
        end
        fprintf('\n');
        n_angle_fft_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
        
    elseif STATIC == 5
        %% 采用APES求解
        fprintf('APES\n');
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1),angleFFTSize);
        
        % 每一个rangebin，逐一进行APES计算
        for i=1:size(radar_data_Static,1)
            radar_data_pre_csl1 = squeeze(radar_data_Static(i,:));
            radar_data_angle_range_Static(i,:) = APES_Alg(radar_data_pre_csl1', angleFFTSize, 40);
        end
        fprintf('\n');
        n_angle_fft_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
        
    elseif STATIC == 6
        %% 采用BF-APES求解
        fprintf('Backward Forward APES\n');
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1),angleFFTSize);
        
        % 每一个rangebin，逐一进行APES计算
        for i=1:size(radar_data_Static,1)
            radar_data_pre_csl1 = squeeze(radar_data_Static(i,:));
            radar_data_angle_range_Static(i,:) = BF_APES_Alg(radar_data_pre_csl1', angleFFTSize, 40);
        end
        fprintf('\n');
        n_angle_fft_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
    elseif STATIC == 7
        %% 采用IAA-APES求解
        fprintf('IAA-APES\n');
        % 输出矩阵初始化
        radar_data_angle_range_Static = zeros(size(radar_data_Static,1),angleFFTSize);
        
        % 每一个rangebin，逐一进行APES计算
        for i=1:size(radar_data_Static,1)
            radar_data_pre_csl1 = squeeze(radar_data_Static(i,:));
            radar_data_angle_range_Static(i,:) = IAA_APES_Alg(radar_data_pre_csl1', angleFFTSize);
        end
        fprintf('\n');
        n_angle_fft_size = size(radar_data_angle_range_Static,2);
        n_range_fft_size = size(radar_data_angle_range_Static,1);
        
    end
    
    if ~DYNAMIC
        %% 不使用MUSIC和其他方法，仅使用FFT计算角度

        %%% 原版程序没有加窗，在这里加个Hanning窗试试，如效果不好可删除 刘明旭 20211026 %%%
        % 为数据加窗
        radar_data_pre_3dfft_hanning = bsxfun(@times, radar_data_pre_3dfft, Angle_Win);
        radar_data_pre_3dfft = radar_data_pre_3dfft_hanning;
        %%% 加窗部分代码结束 20211026 %%%

        % 角度FFT
        radar_data_angle_range = fft(radar_data_pre_3dfft, angleFFTSize, 3);
        n_angle_fft_size = size(radar_data_angle_range,3);
        n_range_fft_size = size(radar_data_angle_range,1);

        %decide non-zerp doppler bins to be used for dynamic range-azimuth heatmap
        % 取3D Cube中，在doppler这一维度上大于阈值的bin，计算R-A Map
        DopplerPower = sum(mean((abs(radar_data_pre_3dfft(:,:,:))),3),1);% 整体取模长；在angle维上取均值（留下range、doppler维度）；在range维上取和，只留doppler维
        DopplerPower_noDC = DopplerPower([1: dopplerFFTSize/2-1 dopplerFFTSize/2+3:end]);% 不取中间（doppler为0附近）的部分bin
        [peakVal peakInd] = max(DopplerPower_noDC); % 取最大值
        threshold = peakVal*ratio;                  %根据最大值设置阈值
        % 取大于阈值的部分bin
        indSel = find(DopplerPower_noDC >threshold);
        for ii = 1:length(indSel)
            if indSel(ii) > dopplerFFTSize/2-1
                indSel(ii) = indSel(ii) + 3;
            end
        end

        % 取不同模式的RangeAngle
        radar_data_angle_range_dynamic = squeeze(sum(abs(radar_data_angle_range(:,indSel,:)),2));           % 运动部分的RangeAngle，在doppler维上取大于阈值部分的模长，累积
%         radar_data_angle_range_Static = squeeze(sum(abs(radar_data_angle_range(:,dopplerFFTSize/2+1,:)),2));% 静止部分的RangeAngle，在速度为0处取模长

        %generate range/angleFFT for zeroDoppler and non-zero Doppler respectively
        % 角度有正负，分别作shiftFFT
        radar_data_angle_range_dynamic = fftshift(radar_data_angle_range_dynamic,2);
%         radar_data_angle_range_Static = fftshift(radar_data_angle_range_Static,2);
        
    elseif DYNAMIC == 1
        %% 使用MUSIC计算动态部分角度  
       
        % 输出矩阵初始化
        radar_data_angle_range_dynamic = zeros(size(radar_data_dynamic,1),angleFFTSize);
        % 取CFAR所得的点数，取每个点的速度、距离，用于计算角度
        numObj = length(detection_obj);        
        range_dynamic = squeeze(zeros(1,size(radar_data_dynamic,1)));% 每个rangebin的目标点数
        range_dynamic_doppIdx = zeros(dopplerFFTSize, size(radar_data_dynamic,1), 'logical');% 每个rangebin对应目标的dopplerbin下标
        for i=1:numObj
            det = detection_obj(i);
            if det.dopplerInd == dopplerFFTSize/2+1
                % 若当前目标点静止，忽略之
                continue;
            end
            range_ind = det.rangeInd+1;
            doppler_ind = det.dopplerInd+1;
            range_dynamic(range_ind) = range_dynamic(range_ind) + 1;
            range_dynamic_doppIdx(doppler_ind, range_ind) = 1;
        end
        
        % 每一个rangebin，逐一进行MUSIC计算
        for i=1:size(radar_data_dynamic,1)
            radar_data_pre_music = squeeze(radar_data_dynamic(i,range_dynamic_doppIdx(:,i),:));
            radar_data_angle_range_dynamic(i,:) = musicAlg(radar_data_pre_music', angleFFTSize, range_dynamic(i));
        end
        % 每一个CFAR所得的点，逐一进行MUSIC计算
        
    end
    
    % 取部分有效的Range下标
    indices_1D = (minRangeBinKeep:n_range_fft_size-rightRangeBinDiscard);
    max_range = (n_range_fft_size-1)*range_resolution;
    max_range = max_range/2;
    d = 1;
    
    sine_theta = -2*((-n_angle_fft_size/2:n_angle_fft_size/2)/n_angle_fft_size)/d;
    cos_theta = sqrt(1-sine_theta.^2);

    % 确定坐标轴
    [R_mat, sine_theta_mat] = meshgrid(indices_1D*range_resolution,sine_theta);
    [~, cos_theta_mat] = meshgrid(indices_1D,cos_theta);
    x_axis = R_mat.*cos_theta_mat;
    y_axis = R_mat.*sine_theta_mat;
    
    % 取有效部分
    mag_data_dynamic = squeeze(abs(radar_data_angle_range_dynamic(indices_1D+1,[1:end 1])));
    mag_data_static = squeeze(abs(radar_data_angle_range_Static(indices_1D+1,[1:end 1])));

    % 转置，翻转
    mag_data_dynamic = mag_data_dynamic';
    mag_data_static = mag_data_static';
%     mag_data_static = 20*log10(mag_data_static);    % 取对数试试
    mag_data_dynamic = flipud(mag_data_dynamic);
    mag_data_static = flipud(mag_data_static);
    
    %%% 取20log10，刘明旭 20211026 %%%
%     mag_data_static = 20.*log10(mag_data_static);
    %%% 结束 20211026 %%%
    
    %%% 每个距离逐一对角度进行归一化 %%%
    % 静态（如果不用CBF，先注释掉）
    for idx_rge = 1:size(mag_data_static, 2)
       temp = mag_data_static(:, idx_rge);
       max_temp = max(temp);
       temp = temp./max_temp;
       temp = temp.*(20*(max_temp^0.6));
%        temp = temp.*(20*log10(max_temp));
       mag_data_static(:, idx_rge) = temp;
    end
    % 动态
%     for idx_rge = 1:size(mag_data_dynamic, 2)
%        temp = mag_data_dynamic(:, idx_rge);
%        max_temp = max(temp);
%        temp = temp./max_temp;
%        temp = temp.*(20*(max_temp^0.2));
% %        temp = temp.*(20*log10(max_temp));
%        mag_data_dynamic(:, idx_rge) = temp;
%     end
%     mag_data_dynamic = 20*log10(mag_data_dynamic);
    %%%结束 20211026 %%%
    
    %%% 观察旁瓣
%     for idx_rge = 1:size(mag_data_static, 2)
%         if idx_rge ~= 63
%            continue 
%         end
%         temp = squeeze(mag_data_static(:, idx_rge));
%         figure(10);
%         plot(temp);
%         xlabel('频率');
%         ylabel('能量');
%         title('金属板所在距离上的信号角度维频谱',idx_rge);
%     end
    %%%

    % 出图
    if PLOT_ON
        log_plot = LOG;
        if STATIC_ONLY == 1
            if log_plot
                surf(y_axis, x_axis, (mag_data_static).^0.4,'EdgeColor','none');%% 显示
            else
                surf(y_axis, x_axis, abs(mag_data_static),'EdgeColor','none');
            end
        else
            if log_plot
                surf(y_axis, x_axis, (mag_data_dynamic).^0.4,'EdgeColor','none');
            else
                surf(y_axis, x_axis, abs(mag_data_dynamic),'EdgeColor','none');
            end
        end

        view(2);
        xlabel('meters-x')
        ylabel('meters-y')
        title('range/azimuth heat map static objects')
        
        %%% 角度显示，临时用 %%%
        figure(3);                
        axis_range = indices_1D*range_resolution;
        axis_angle = asin(sine_theta).*(180/pi);
        [X,Y] = meshgrid(axis_range,axis_angle);
%         mesh(Y',X',((mag_data_static).^0.4)');
        mesh((mag_data_static'));
        xlabel('angle-degree');        
        ylabel('range-meter');        
        title('Range-Angle Map');
        view(2);
        %%%

    end
    
end