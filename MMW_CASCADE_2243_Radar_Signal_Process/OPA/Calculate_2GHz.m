clc;clear;       % 位宽计算、绝对精度、线性度差到什么程度不能用
close all;

% 调参区域
fibacktarget = 'longshao_002';      % 信号文件名 33-245m_001  60mfibrt-1998MHz_000
fifront = 'D:\radardata\OPA\20220414\';        % 触发信号文件位置
Vc = 3e8;
T = 1e-5;
B = 1.9984e9;
lamda0 = 1.55e-6;

% 读取数据
fileback1 = '.mat';
filename2 = [fifront,fibacktarget,fileback1];
fileread = load(filename2);
samplingtime1 = fileread.sampleInterval;
fs = 1/samplingtime1;
normal = fileread.data;

Y = normal';
len = length(Y);
t = (1:1:len)/fs;
Y = fftfilter2(Y,fs,14e6,10e7);
figure,subplot(2,1,1),plot(t,Y);
xlabel('Time / s');
ylabel('Intencity / cd');
title('Input Signal');

[real0,freq0] = my_fft(Y,fs);           % 傅里叶变换
subplot(2,1,2),plot(freq0,real0);
title('FFT Transform-All');
xlabel('Frequence / Hz');
ylabel('Intencity / cd');

% [Amp,Phi,Fre] = MY_HHT(Y,fs);           % 希尔伯特频率计算
% figure,subplot(2,1,1),plot(t,abs(Fre));
% title('Time-Frequence Curve');
% xlabel('Time / s');
% ylabel('Frequence / Hz');
% 
% F1 = smooth(abs(Fre),80,'sgolay');      % 时频滤波
% subplot(2,1,2),plot(t,F1);
% title('Time-Frequence-Filter Curve');
% xlabel('Time / s');
% ylabel('Frequencye/ Hz');

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
result_reaz = zeros(round(cut_long/2),round(len/cut_long)); % FFT结果（对应点位的幅值）——20220414
result_frez = zeros(round(cut_long/2),round(len/cut_long)); % FFT结果（对应点位的频率）——20220414
ti = 5;
nnn = 950:1000;
% figure;
for n = 1:data_long
    datai = reY(:,n).';
    [reaz0,frez] = my_fft(datai,fs);    
    reaz = smooth(reaz0.^2,6,'moving')';
    result_reaz(:,n) = reaz;    % 这里保留平滑后的幅值——20220414
    result_frez(:,n) = frez;    % 保留频点——20220414
%     reaz = reaz0.^2;
    [fftfreq1,fftfreq2] = myselect(reaz,frez,17,2);  % 查找两个峰值
    pos1 = find(frez == fftfreq1);
    pos2 = find(frez == fftfreq2);
    freq_center0 = abs(fftfreq1+fftfreq2)/2;
    V_fft = abs(fftfreq1-fftfreq2)*lamda0/4;
    R_fft = freq_center0*Vc*T/B/4;
    result1(n) = R_fft;
    
    % 重心法
    [freq_cent1,freq_cent2] = my_center(frez,reaz,pos1,pos2,5); % 重心法查找两个峰值
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

figure,subplot(2,1,1),plot(frez,reaz0);
title('Before filtering');
xlabel('Frequence / Hz');
ylabel('Intencity / cd');

subplot(2,1,2),plot(frez,reaz);
title('After filtering');
xlabel('Frequence / Hz');
ylabel('Intencity / cd');

figure,subplot(2,1,1),plot(result0,'r');
xlabel('Times / Hz');
ylabel('Distance / m');
title('Long Time Ranging Accurancy');
subplot(2,1,2),plot(V0,'r');
xlabel('Times / Hz');
ylabel('Velocity / m');
title('Long Time Velocity Accurancy');

figure,plot(result0,'r');
xlabel('Times / Hz');
ylabel('Distance / m');
title('Long Time Ranging Accurancy');
