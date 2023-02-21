function [outFreqSignal] = BF_APES_Alg(signal,L,M)
    %% Backward Forward APES（前后向正弦振幅相位估计,Amplitude and Phase Estimation of Sinusoid）算法
    %   输入1维信号，计算波束形成
    %   signal: 1维复数时域信号，格式是channel×n
    %   L：扫描角度向量
    %   M：FIR滤波器阶数
    
    lamda = 1;
    d=lamda/2;
    [N, objs] = size(signal);
    
    % 默认参数赋值
    if ~exist('M','var')
        M = int(N / 2);
    end
    
    
    % N:天线个数
    if objs > 0
        %% 有数据才有计算角度的必要性
        %%% BF APES中，需要把数据反褶共轭，计算后向部分 %%%
        %%% 然后和前向部分放在一起，计算出最终结果 %%%
        
        %%% 计算反褶共轭
        signal_b = flip(signal, 1);
        signal_b = conj(signal_b);
        
        %%% 计算样本相关矩阵
        % 论文里的公式（普通APES）：
        % R = [1 / (N - M + 1)] Σ(z_i * z_i_H)
        % 这里把z_i按列拼到一起了，xs = [z_0, z_1, ..., z_N-M]，这样就可以直接转置相乘得到R
        % 前向部分
        R_f = zeros(M);
        for k=1:N-M                          
%             xs(:,k)=signal(k+M-1:-1:k);
            z_i = signal(k+M-1:-1:k,:);
            R_f = R_f + z_i * z_i';
        end
        R_f = R_f / (N - M);
        % 后向部分
        R_b = zeros(M);
        for k=1:N-M                          
%             xs(:,k)=signal(k+M-1:-1:k);
            z_i_b = signal_b(k+M-1:-1:k,:);
            R_b = R_b + z_i_b * z_i_b';
        end
        R_b = R_b / (N - M);
        R = (R_f + R_b) / 2;
        R_inv = inv(R);
        
        %%% 扫描所有角度，计算波束形成
        antennaArr = linspace(0,(N-1)*d,N)';
        sine_theta = -2*((-L/2:L/2)/L)/(2*d);
        L_theta = asin(sine_theta);
        % scanAngle = linspace(-pi/2,pi/2,L)';
        scanAngle = L_theta';
        powerSpectrumInSpace = zeros(1,L);
        for i =1:L
            theta = scanAngle(i);
            % 计算g(w)，也就是z_i的归一化DFT
            % 前向
            gw_f = zeros(M, objs);
            for l=1:N-M
                z_i = signal(l+M-1:-1:l,:);
                gw_i = z_i * exp(-1i*pi*sin(theta)*(l+M-1));
                gw_f = gw_f + gw_i;
            end
            gw_f = gw_f / (N - M);
            % 后向
            gw_b = zeros(M, objs);
            for l=1:N-M
                z_i_b = signal_b(l+M-1:-1:l,:);
                gw_i_b = z_i_b * exp(-1i*pi*sin(theta)*(l+M-1));
                gw_b = gw_b + gw_i_b;
            end
            gw_b = gw_b / (N - M);
            gw = [gw_f, gw_b] / sqrt(2);
            
%             Q=R-gw*gw';                      % 计算矩阵Q
            Q_inv = R_inv - R_inv * gw * inv(gw' * R_inv * gw - eye(size(gw, 2))) * gw' * R_inv;
            for l=1:M
                aw(l,:)=exp(-1i*pi*sin(theta)*(l-1));        % 计算a(w)
            end
            A1=(aw'* Q_inv * gw)/(aw' * Q_inv * aw);
%             A(:,i) = A1;                           % 计算幅度谱
            
            
%             av = array_response_vector(antennaArr,theta);
            powerSpectrumInSpace(i) = abs(A1 * A1');
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
