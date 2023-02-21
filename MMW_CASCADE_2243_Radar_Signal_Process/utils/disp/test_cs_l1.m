%function Farfield_DOA (resignal)
% 这里是基于声波模型做的仿真，和radar的还是有所区别
clc;
clear all;
close all;

% 参数设置
Freq =700;                    %Frequency
snr =30;                      %Source to noise ratio
M =86;                          %The number of microphones（麦克风数目，应该是可以理解为天线数）
Cspeed = 340;                    % The speed of sound
Lambda = Cspeed/340;             %Wavelength
d = Lambda/2;                     %Array elements spacing
% Dirsources = [-10,20,35,50];             %Direction of sound sources
Dirsources = [-80, -60, -30, -10, 0, 5, 15, 20, 50, 70];             %Direction of sound sources
Numsources = length(Dirsources);      %Number of sound sources

% 信号生成
for q = 1:Numsources
    a(:,q)=exp(-1i * 2 * pi * d *(0:M-1)'*sin(Dirsources(q)*pi/180)/Lambda);
end                             % a矩阵，[M*Numsources],来自每个信号源，作用在每个天线上
Energy = [1,200,30,2,40000,6,10,3,30,5];                 %Sources energy 信号源的能量级
y = a*Energy';                  %Received signal of array 将来自不同信号源的信号叠加，y[M*1]
Y = awgn(y,snr,'measured');     %Adding noise 添加噪声

% 扫描方向
grid = 0.1;                              %The value of grid node  
dirmin = -90;                            %The min and max DOA value of the source plane
dirmax = 90;
Dirmicrogrid = dirmin:grid:dirmax;       %Direction between microphone and grid node
L = 1801;
antennaArr = linspace(0,(L-1)*d,L)';
sine_theta = -2*((-L/2:L/2)/L)/(2*d);
L_theta = asin(sine_theta);
% scanAngle = linspace(-pi/2,pi/2,L)';
scanAngle = L_theta' .*(180/pi);
Numgrid = length(scanAngle);          %Number of grid


% 优化求解
tic
resignal = CS_L1Alg(Y,L) ;       %Recover using OMP
resignal = abs(resignal');
toc
 
nonsource = nonzeros(resignal);
 
figure(1);
 
plot(scanAngle(1:L),resignal,'*-')%  CS得到的信号功率谱
hold on
plot(-Dirsources,Energy,'rs')    % 信号真值
legend('CS得到的信号功率谱','信号真值');
title(['SNR = ',num2str(snr)])
xlabel('信源方向角')
ylabel('信源幅度值')
dirction = (find(resignal) -1) *grid -90;