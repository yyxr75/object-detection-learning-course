function [outFreqSignal] = CaponAlg(signal,L)
    %% Capon（自适应波束形成,MVDR）算法
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
        % 计算correlation matrix（由于采样长度有限，只能用接收信号的covariance matrix来替代）
        X = (signal*signal') / objs;% '这是共轭转置
        
        % 对角加载，增加稳健性
        eigVal = complex(zeros(N),zeros(N));
        [eigVec,eigVal] = eig(X);
        lambda = abs(eigVal(1,1)) * 10;
        lambda_I = eye(size(X,1)) * lambda;
%         X = X + lambda_I;
        
        X_inv = inv(X);
        %%% 扫描所有角度，计算波束形成
        antennaArr = linspace(0,(N-1)*d,N)';
        sine_theta = 2*((-L/2:L/2)/L)/(2*d);
        L_theta = asin(sine_theta);
        % scanAngle = linspace(-pi/2,pi/2,L)';
        scanAngle = L_theta';
        powerSpectrumInSpace = zeros(1,L);
        for i =1:L
            av = array_response_vector(antennaArr,scanAngle(i));
            powerSpectrumInSpace(i) = 1/(av'*X_inv*av);
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
