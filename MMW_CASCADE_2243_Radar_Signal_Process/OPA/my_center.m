% 重心法频谱细化
% 输出两个峰值的频率
% 输入依次为（横坐标，纵坐标，第一峰值位置，第二峰值位置，拟合半宽）

function [F1,F2] = my_center(fre,rea,pos1,pos2,bw)

cut_fre1 = fre(pos1-bw:pos1+bw);
cut_fre2 = fre(pos2-bw:pos2+bw);
cut_rea1 = rea(pos1-bw:pos1+bw);
cut_rea2 = rea(pos2-bw:pos2+bw);

cut_rea1 = cut_rea1-min(cut_rea1);
cut_rea2 = cut_rea2-min(cut_rea2);

% cut_rea1 = cut_rea1(1:end-2).*cut_rea1(2:end-1).*cut_rea1(3:end);
% cut_rea2 = cut_rea2(1:end-2).*cut_rea2(2:end-1).*cut_rea2(3:end);

F1 = sum(cut_fre1.*cut_rea1)/sum(cut_rea1);
F2 = sum(cut_fre2.*cut_rea2)/sum(cut_rea2);




