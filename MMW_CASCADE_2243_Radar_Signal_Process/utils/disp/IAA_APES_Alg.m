function [outFreqSignal] = IAA_APES_Alg(signal,L)
    %% IAA APES（迭代自适应正弦振幅相位估计,Iterative Adaptive Approach for Amplitude and Phase Estimation of Sinusoid）算法
    %   输入1维信号，计算波束形成
    %   signal: 1维复数时域信号，格式是channel×n
    %   L：扫描角度向量
    
    lamda = 1;
    d=lamda/2;
    [N, objs] = size(signal);    
    
    % N:    天线个数
    % objs: 快拍数
    if objs > 0
        %% 有数据才有计算角度的必要性
        %%% 初始化
        antennaArr = linspace(0,(N-1)*d,N)';
        sine_theta = -2*((-L/2:L/2)/L)/(2*d);
        L_theta = asin(sine_theta);
        % scanAngle = linspace(-pi/2,pi/2,L)';
        scanAngle = L_theta';
        % 方向矩阵
        a = zeros(N, L, 'double');
        A = complex(a, 0);                  % 稀疏方向矩阵
        for i = 1:L
            % 为方向矩阵赋值
           A(:, i) = array_response_vector(antennaArr,scanAngle(i));
        end
        % 功率谱
        powerSpectrumInSpace = zeros(1,L);
        % 功率谱矩阵初始化
        P = eye(L);
        for i = i:L
           A_i = A(:, i);
           P_i = A_i' * signal;
           P_i = (norm(P_i * P_i')^2) / ((A_i' * A_i)^2 * N);
           P(i, i) = P_i;
        end
        
        % IAA迭代
        err_last = -1.;
        while true
            %%% 计算协方差矩阵R
            R = A * P * A';
            R_inv = inv(R);
            %%% 初始化迭代方差、当前的功率谱
            powerSpectrumInSpace_i = zeros(1,L);
            err = 0.;
            %%% 扫描所有角度，估计功率谱
            for i =1:L
                A_i = A(:, i);
                s_i = (A_i' * R_inv * signal) / (A_i' * R_inv * A_i);
                P_i = norm(s_i)^2 / objs;
                err = err + abs(P(i, i) - P_i);
                P(i, i) = P_i;
                powerSpectrumInSpace(i) = P_i;
            end
            
            %%% 根据残差判断是否结束迭代
%             fprintf("err of IAA: %f.\n", err);
            if err < err_last
                break;
            else
                err_last = err;
%                 powerSpectrumInSpace = powerSpectrumInSpace_i;
            end
        end 
        
        
            % 线性映射
    %         figure(5);
    %         plot(powerSpectrumInSpace);
    %         hold on;
    %         powerSpectrumInSpace = (powerSpectrumInSpace-min(powerSpectrumInSpace))/(max(powerSpectrumInSpace)-min(powerSpectrumInSpace));
            % outFreqSignal = powerSpectrumInSpace/max(powerSpectrumInSpace);
    %         powerSpectrumInSpace = powerSpectrumInSpace / min(powerSpectrumInSpace);


    else
        %% 否则，直接返回就行了
        powerSpectrumInSpace = ones(1,L);
    end

    outFreqSignal = powerSpectrumInSpace;

end
