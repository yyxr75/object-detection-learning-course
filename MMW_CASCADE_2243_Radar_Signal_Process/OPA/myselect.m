function [peak1,peak2] = myselect(absy,hz,ti,sub)

py = absy;   % 防止原始值被覆盖
pos = find(py == max(py));
py(1:ti+1) = 0;
py(end-ti:end) = 0;
if length(pos) == 1
    py(pos-ti:pos+ti) = 0;
    pos1 = pos;
    pos2 = find(py == max(py));
else
    pos2 = pos(2);
    pos1 = pos(1);
end

if absy(pos1) >sub*py(pos2) || abs(pos1-pos2)<ti
    pos2 = pos1;
end

cot = py(pos2)>=py(pos2-ti:pos2+ti);
coe = find(cot==0);

if ~isempty(coe)
    pos2 = pos1;
%     m=0;
end

if max(absy)<10*mean(absy)
    peak1 = NaN;
    peak2 = NaN;
else
    peak1 = hz(pos1);
    peak2 = hz(pos2);
end
