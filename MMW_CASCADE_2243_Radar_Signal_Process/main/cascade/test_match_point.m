% 尝试下两个部分的频谱怎么匹配到一起（仅限于lidar单点）
clc;clear;       
close all;

% 参数初始化
Vc = 3e8;
T = 1e-5;
B = 1.9984e9;
lamda0 = 1.55e-6;

% 读取毫米波雷达数据
file_name_ = '581';      % 
file_path_ = 'D:\radardata\OPA\radar\';        % 触发信号文件位置
file_back = '.mat';
filename_radar = [file_path_,file_name_,file_back];
fileread_radar = load(filename_radar);
axis_range_radar = fileread_radar.axis_range;
radar_data = fileread_radar.ss;
radar_data = radar_data / max(radar_data);

% 读取OPA雷达数据
fibacktarget = '2-514m_000';      % 信号文件名 33-245m_001  60mfibrt-1998MHz_000
fifront = 'D:\radardata\OPA\';        % 触发信号文件位置
fileback1 = '.mat';
filename2 = [fifront,fibacktarget,fileback1];
fileread = load(filename2);
samplingtime1 = fileread.sampleInterval;
fs = 1/samplingtime1;
normal = fileread.data;

Y = normal';
len = length(Y);
t = (1:1:len)/fs;
Y = fftfilter2(Y,fs,9e6,50e7);
figure,subplot(2,1,1),plot(t,Y);
xlabel('Time / s');
ylabel('Intencity / cd');
title('Input Signal');

[real0,freq0] = my_fft(Y,fs);           % 傅里叶变换
subplot(2,1,2),plot(freq0,real0);
title('FFT Transform-All');
xlabel('Frequence / Hz');
ylabel('Intencity / cd');

[Amp,Phi,Fre] = MY_HHT(Y,fs);           % 希尔伯特频率计算
figure,subplot(2,1,1),plot(t,abs(Fre));
title('Time-Frequence Curve');
xlabel('Time / s');
ylabel('Frequence / Hz');

F1 = smooth(abs(Fre),80,'sgolay');      % 时频滤波
subplot(2,1,2),plot(t,F1);
title('Time-Frequence-Filter Curve');
xlabel('Time / s');
ylabel('Frequencye/ Hz');

cut_time = T;
cut_long = cut_time*fs;         % 每周期采样点数
data_long = round(len/cut_long);% 总周期数
% data_long = 73;
cut_t = (1:1:cut_long);         % 时间轴
reY = reshape(Y,round(cut_long),round(len/cut_long));% 将数据拆分为每周期采样*周期数
V0 = zeros(1,data_long);
result0 = zeros(1,data_long);
result1 = zeros(1,data_long);
Peak0 = zeros(1,data_long);
ti = 5;
nnn = 950:1000;
% figure;
for n = 1:data_long
    datai = reY(:,n).';
    [reaz0,frez] = my_fft(datai,fs);    
    reaz = smooth(reaz0.^2,3,'moving')'; 
%     reaz = reaz0.^2;
    [fftfreq1,fftfreq2] = myselect(reaz,frez,7,2);  % 查找两个峰值
    pos1 = find(frez == fftfreq1);
    pos2 = find(frez == fftfreq2);
    freq_center0 = abs(fftfreq1+fftfreq2)/2;
    V_fft = abs(fftfreq1-fftfreq2)*lamda0/4;
    R_fft = freq_center0*Vc*T/B/4;
    result1(n) = R_fft;
    
    % 重心法
    [freq_cent1,freq_cent2] = my_center(frez,reaz,pos1,pos2,3); % 重心法查找两个峰值
    freq_center = abs(freq_cent1+freq_cent2)/2;
    V_center = abs(freq_cent1-freq_cent2)*lamda0/4;
    R_center = freq_center*Vc*T/B/4;
    Peak0(n) = freq_center;
    result0(n) = R_center;
    V0(n) = V_center;
%     plot(frez,10*log10(reaz0));  
%     hold on,stem(freq_center,max(reaz))
%     plot(frez,reaz,'g');
%     hold off;
end

figure;
subplot(2,1,1);
plot(frez*Vc*T/B/4,reaz0);
title('Before filtering');
xlabel('Frequence / Hz');
ylabel('Intencity / cd');
hold on;
% 毫米波雷达数据
radar_data = radar_data * max(reaz0);
plot(axis_range_radar, radar_data);
hold off;

subplot(2,1,2),plot(frez,reaz);
title('After filtering');
xlabel('Frequence / Hz');
ylabel('Intencity / cd');

% figure,subplot(2,1,1),plot(result0,'r');
% xlabel('Times / Hz');
% ylabel('Distance / m');
% title('Long Time Ranging Accurancy');
% subplot(2,1,2),plot(V0,'r');
% xlabel('Times / Hz');
% ylabel('Velocity / m');
% title('Long Time Velocity Accurancy');

figure,subplot(2,1,1),plot(result0,'r');
xlabel('Times / Hz');
ylabel('Distance / m');
title('Long Time Ranging Accurancy');

subplot(2,1,2),plot(result1,'b');
xlabel('Times / Hz');
ylabel('Distance / m');
title('Long Time Ranging Accurancy');

dis = mean(result0)-3.3
Uncertenity = 3*std(result0)



figure,plot(result0,'r');
xlabel('Times / Hz');
ylabel('Distance / m');
title('Long Time Ranging Accurancy');
