% 读取保存的数据，尝试多帧的Capon Beamforming
clc;
close all;
clear;

FRAME_BEGIN = 2;        % 起始帧号
FRAME_END = 106;          % 结束帧号
angleFFTSize = 512;     % 角度分辨力
minRangeBinKeep = 5;
rightRangeBinDiscard = 20;
range_resolution = 0.0593;

%%% 读取数据
for idx = FRAME_BEGIN:FRAME_END
    data_name = ['D:\radar\cascade_demo\4chip_cascade_MIMO_example\save_data\capon_test\', num2str(idx), '.mat'];    
	signal(idx) =  load(data_name);
end

%%% 每个距离bin分别采用Capon进行多帧计算
radar_data_angle_range_Static = zeros(size(signal(FRAME_BEGIN).radar_data_Static, 1), angleFFTSize);  % 输出矩阵初始化
for i=1:size(signal(FRAME_BEGIN).radar_data_Static, 1)
    %% 每个距离bin分别采用Capon进行多帧计算
    for frame_id = FRAME_BEGIN:FRAME_END
        %% 逐帧读取对应的数据
        radar_data_Static_ = signal(frame_id).radar_data_Static;
        radar_data_pre_capon(frame_id - FRAME_BEGIN + 1, :) = squeeze(radar_data_Static_(i,:));
        
    end
    % Capon计算角度
    radar_data_angle_range_Static(i,:) = CaponAlg(radar_data_pre_capon', angleFFTSize);
end
n_angle_fft_size = size(radar_data_angle_range_Static,2);
n_range_fft_size = size(radar_data_angle_range_Static,1);

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
mag_data_static = squeeze(abs(radar_data_angle_range_Static(indices_1D+1,[1:end 1])));
% 转置，翻转
mag_data_static = mag_data_static';
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

%%% 出图
figure(1);      % 图1，range-angle图和x-y图
subplot(121);   % 1子图，range-angle图
axis_range = indices_1D*range_resolution;
axis_angle = asin(sine_theta).*(180/pi);
[X,Y] = meshgrid(axis_range,axis_angle);
mesh(X,Y,(mag_data_static).^0.4);
% mesh((mag_data_static));
xlabel('range');
ylabel('angle');
subplot(122);   % 2子图，x-y图
surf(y_axis, x_axis, (mag_data_static).^0.4,'EdgeColor','none');
xlabel('meters');    ylabel('meters');