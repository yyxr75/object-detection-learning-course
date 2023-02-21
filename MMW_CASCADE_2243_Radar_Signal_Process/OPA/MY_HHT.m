function [Amp,Phi,Fre] = MY_HHT(line_y,fs)
% 依次输入：函数，函数坐标；
% 依次输出：幅值，相位，频率。
% 希尔伯特变换

lin = line_y-mean(line_y);
h_lin = hilbert(lin);
Amp = abs(h_lin);
Phi = angle(h_lin);
Phi = unwrap(Phi);
Fre = diff(Phi)*fs/2/pi;
Fre = [Fre(1) Fre];
