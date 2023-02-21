% 尝试下两个部分的频谱怎么匹配到一起（LiDAR单线，Radar静态）
clear;
clc;
close all;

% 参数初始化
Vc = 3e8;
T = 1e-5;
B = 1.9984e9;
lamda0 = 1.55e-6;

% 读取毫米波雷达数据
file_name_radar = 'radar_ren';      % 文件名
file_path_radar = 'D:\radar\cascade_demo\4chip_cascade_MIMO_example\save_data\match_test\20220414\';        % 文件位置
file_back_radar = '.mat';
filename_radar = [file_path_radar,file_name_radar,file_back_radar];
fileread_radar = load(filename_radar);
axis_range_radar = fileread_radar.axis_range;
radar_data = fileread_radar.ss;
radar_data = radar_data / max(radar_data);