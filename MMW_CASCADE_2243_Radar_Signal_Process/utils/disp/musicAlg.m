function [outFreqSignal] = musicAlg(signal,L,D)
%MUSICALG 自适应music算法
%   输入1维信号，按照thresh分割特征向量和特征值，输出频率分析结果
%   signal: 1维复数时域信号，格式是channel×n
%   L：扫描角度向量
%   D: 目标点树目
    
    % 默认参数赋值
    if ~exist('D','var')
        D = 0;
    end
    
    lamda = 1;
    d=lamda/2;
    [M, objs] = size(signal);
    % filename1 = ['rawsig_',datestr(now,'HHMMSSFFF'),'.mat'];
    % save(filename1,'signal_1d');
    % L = 360;
    % N:天线个数
    N=M;
    %% 有数据才有计算MUSIC的必要性
    if objs > 0
        % 计算correlation matrix（由于采样长度有限，只能用接收信号的covariance matrix来替代）
        % 按照特征向量变化的一阶导来划分信号空间和噪声空间
        X = signal*signal';% '这是共轭转置
        % filename2 = ['covmat_',datestr(now,'HHMMSSFFF'),'.mat'];
        % save(filename2,'X');
        [eigVec,eigVal] = eig(X);
        eigVal = diag(eigVal);
    %     figure(5);
    %     plot(eigVal,'*');
    %     hold on;
        % 以最大特征值的1/10为阈值，小于的就视为噪声子空间对应的特征值
        eig_max = max(eigVal);
        eig_min = min(eigVal);
        eig_n_idx = []; % 噪声子空间对应的特征值
        if abs(eig_max/eig_min) > 100 && D > 0
            % 特征值之间的差值大于100，才认为存在信号子空间，计算MUSIC
            for i = 1:size(eigVal)
                if eigVal(i) < eig_max * 0.1
                    eig_n_idx = [eig_n_idx, i];
                end
            end
            % 尽量采用更多的目标数目
            if size(eig_n_idx, 2) > M-D
                eig_n_idx = 1:M-D;
            end
%             figure(10)
%             plot(eigVal,'b*');
%             hold on;
%             plot(eig_n_idx, eigVal(eig_n_idx), 'r*');
%             hold off;
            %
            Qn = eigVec(:,eig_n_idx);
            %% 扫描所有角度，计算谱分析结果
            antennaArr = linspace(0,(N-1)*d,N)';
            sine_theta = -2*((-L/2:L/2)/L)/(2*d);
            L_theta = asin(sine_theta);
            % scanAngle = linspace(-pi/2,pi/2,L)';
            scanAngle = L_theta';
            powerSpectrumInSpace = zeros(1,L);
            for i =1:L
                av = array_response_vector(antennaArr,scanAngle(i));
                powerSpectrumInSpace(i) = 1/norm(Qn'*av);
%                 if powerSpectrumInSpace(i) < 1.1% 经验阈值，临时策略
%                     powerSpectrumInSpace(i) = 1;
%                 end
            end
            % 线性映射
    %         figure(5);
    %         plot(powerSpectrumInSpace);
    %         hold on;
    %         powerSpectrumInSpace = (powerSpectrumInSpace-min(powerSpectrumInSpace))/(max(powerSpectrumInSpace)-min(powerSpectrumInSpace));
            % outFreqSignal = powerSpectrumInSpace/max(powerSpectrumInSpace);
    %         powerSpectrumInSpace = powerSpectrumInSpace / min(powerSpectrumInSpace);

        else
            % 否则，认为不存在信号子空间，只有噪声子空间
            powerSpectrumInSpace = ones(1,L);
        end
    else
        % 否则，直接返回就行了
        powerSpectrumInSpace = ones(1,L);
    end

    outFreqSignal = powerSpectrumInSpace;

end
