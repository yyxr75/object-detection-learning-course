function [outFreqSignal] = CS_L1Alg(signal,L)
    %% Compressed Sensing L1（基于噪声约束和L1范数优化的压缩感知）算法
    %   输入1维信号，计算波束形成
    %   signal: 1维复数时域信号，格式是channel×n
    %   L：扫描角度向量
    
    lamda = 1;
    d=lamda/2;
    [M, objs] = size(signal);
    % N:天线个数
    N=M;
    if objs > 0
        %% 有数据才有计算角度的必要性
        
        % 计算噪声功率
        snr = 100000;
        pwr_noise = sqrt(abs(signal' * signal)) / snr;
%         fprintf('%f\n',pwr_noise);
%         if pwr_noise > 1000 % 把残差功率限制在1000以下
%             pwr_noise = 1000;
%         end
        
        %%% 扫描所有角度，计算压缩感知
        antennaArr = linspace(0,(N-1)*d,N)';
        sine_theta = -2*((-L/2:L/2)/L)/(2*d);
        L_theta = asin(sine_theta);
        % scanAngle = linspace(-pi/2,pi/2,L)';
        scanAngle = L_theta';
        
        powerSpectrumInSpace = zeros(1,L);  % 最终输出的功率谱
        a = zeros(N, L, 'double');
        A = complex(a, 0);                  % 稀疏方向矩阵
        for i = 1:L
            % 为方向矩阵赋值
           A(:, i) = array_response_vector(antennaArr,scanAngle(i));
        end
        
        %%% 基于cvx优化求解稀疏解s %%%
        cvx_begin quiet
        variable coefficients(L) complex
        minimize(norm(coefficients, 1))
        subject to 
        norm(signal - A * coefficients) <= pwr_noise;
        cvx_end
        
        
        
        % 迭代完成后，为空间谱赋值
        powerSpectrumInSpace = coefficients;
        

        
            % 线性映射
    %         figure(5);
    %         plot(powerSpectrumInSpace);
    %         hold on;
    %         powerSpectrumInSpace = (powerSpectrumInSpace-min(powerSpectrumInSpace))/(max(powerSpectrumInSpace)-min(powerSpectrumInSpace));
            % outFreqSignal = powerSpectrumInSpace/max(powerSpectrumInSpace);
    %         powerSpectrumInSpace = powerSpectrumInSpace / min(powerSpectrumInSpace);


    else
        %% 否则，直接返回就行了
        powerSpectrumInSpace = zeros(1,L);
    end

    outFreqSignal = powerSpectrumInSpace;

end
