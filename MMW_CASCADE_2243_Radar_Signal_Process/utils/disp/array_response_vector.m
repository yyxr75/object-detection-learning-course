%  Calculate the steering vector for a certain angle, theta.
%  所谓steering vector就是目标在某个角度下的对每根天线形成的相位差序列
function [steerV] = array_response_vector(array, theta)
    [N,~] = size(array);
%     # Calculate the steering vector, v for certain angle, theta. Shape of v is N by 1. See equation (5) in [1].
    v = exp(1j*2*pi*array*sin(theta));
%     """ print('The steering vector for certain angle %f is: ' % (theta*180/np.pi), v) """
    steerV = v/sqrt(N);
end