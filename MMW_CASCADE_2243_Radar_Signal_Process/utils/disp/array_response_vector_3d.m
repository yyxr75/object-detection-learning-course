%  Calculate the steering vector for a certain angle, theta.
%  所谓steering vector就是目标在某个角度下的对每根天线形成的相位差序列
function [steerV] = array_response_vector_3d(array, theta, phi)
    [N,~] = size(array);
%     # Calculate the steering vector, v for certain angle, theta. Shape of v is N by 1. See equation (5) in [1].
%     v = exp(1j * pi * (array(:,1)*sin(theta*pi/180) + array(:,2)*sin(phi*pi/180)) );% 论文里是sin(theta)cos(phi)*x + sin(phi)*y
    v = exp(1j * pi * (array(:,1)*sin(theta*pi/180)*cos(phi*pi/180) + array(:,2)*sin(phi*pi/180)) );% 论文里是sin(theta)cos(phi)*x + sin(phi)*y
    steerV = v/sqrt(N);
end