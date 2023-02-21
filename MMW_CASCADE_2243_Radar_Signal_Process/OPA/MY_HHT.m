function [Amp,Phi,Fre] = MY_HHT(line_y,fs)
% �������룺�������������ꣻ
% �����������ֵ����λ��Ƶ�ʡ�
% ϣ�����ر任

lin = line_y-mean(line_y);
h_lin = hilbert(lin);
Amp = abs(h_lin);
Phi = angle(h_lin);
Phi = unwrap(Phi);
Fre = diff(Phi)*fs/2/pi;
Fre = [Fre(1) Fre];
