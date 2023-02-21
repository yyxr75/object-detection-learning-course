function [rea,hz] = my_fft(fft_y,fs)
%输入：需要进行变换的向量和对应的最大频率；
%输出：傅里叶变换的幅值，对应每个点位的频率。

lenfft = length(fft_y);
fft_fft_y = fft(fft_y);
abs_fft = abs(fft_fft_y);

% pha_fft = phase(fft_fft_y);
abs_fft(1) = 0;
rea1 = 2*abs_fft/(lenfft/2);    % 还原真实的幅值
rea = rea1(1:round(end/2));
% img = pha_fft;
hz1 = (0:1:lenfft-1)*fs/lenfft; % FFT之后每个点位对应的频率
hz = hz1(1:round(end/2));
