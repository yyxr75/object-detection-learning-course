function [rea,hz] = my_fft(fft_y,fs)
%���룺��Ҫ���б任�������Ͷ�Ӧ�����Ƶ�ʣ�
%���������Ҷ�任�ķ�ֵ����Ӧÿ����λ��Ƶ�ʡ�

lenfft = length(fft_y);
fft_fft_y = fft(fft_y);
abs_fft = abs(fft_fft_y);

% pha_fft = phase(fft_fft_y);
abs_fft(1) = 0;
rea1 = 2*abs_fft/(lenfft/2);    % ��ԭ��ʵ�ķ�ֵ
rea = rea1(1:round(end/2));
% img = pha_fft;
hz1 = (0:1:lenfft-1)*fs/lenfft; % FFT֮��ÿ����λ��Ӧ��Ƶ��
hz = hz1(1:round(end/2));
