function [outFreqSignal] = CS_OmpAlg(signal,L)
    %% Compressed Sensing OMP（正交匹配追踪的压缩感知）算法
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
        
        %%% 扫描所有角度，计算压缩感知
        antennaArr = linspace(0,(N-1)*d,N)';
        sine_theta = -2*((-L/2:L/2)/L)/(2*d);
        L_theta = asin(sine_theta);
        % scanAngle = linspace(-pi/2,pi/2,L)';
        scanAngle = L_theta';
        
        powerSpectrumInSpace = zeros(1,L);  % 最终输出的功率谱
        products = zeros(1,L);              % 每次迭代的功率谱
        a = zeros(N, L, 'double');
        A = complex(a, 0);                  % 稀疏方向矩阵
        for i = 1:L
            % 为方向矩阵赋值
           A(:, i) = array_response_vector(antennaArr,scanAngle(i));
        end
        r_n = signal;       % 信号残差，初值与输入信号相同
        Aug_t = [];         %已寻找到目标的方位矢量
        times = 0;          % 已找到的目标数
        pos_arrapy = [];    % 已找到的目标位置
        aug_x = [];         % 已找到的目标能量
        
        % 计算噪声功率
        snr = 10000;
        pwr_noise = sqrt(abs(signal' * signal)) / snr;
%         fprintf('%f\n',pwr_noise);
%         if pwr_noise > 1000 % 把残差功率限制在1000以下
%             pwr_noise = 1000;
%         end
        
        while norm(r_n) > pwr_noise
            %% OMP迭代，退出条件为，信号残差的能量小于阈值
            % 计算残差部分的功率谱（采用CBF计算）
            CovMat_rn = r_n * r_n';  % 自相关矩阵
            for i =1:L
                av = array_response_vector(antennaArr,scanAngle(i));
                products(i) = av'*CovMat_rn*av;
            end
            % 寻找功率谱函数的顶点
            [maxv,maxl]=findpeaks(abs(products),'minpeakdistance',1,'minpeakheight',1);
            % 若没有找到峰值，直接退出
            if size(maxv, 2) < 1
               break
            end
            % 确定顶点位置
            [va, lo] = max(maxv);   % va-顶点的值；lo-顶点在maxv中的位置
            pos = maxl(lo);         % 顶点在功率谱中的位置
            % 为防止重复，检测稀疏方向矩阵对应的列是否为0
            while norm(A(:, pos)) == 0
                maxv(pos) = [];
                maxl(pos) = []; % 删除重复的顶点
                % 若删除后峰值序列为空，则退出循环
                if size(maxv, 2) < 1
                    pos = -1;
                    break
                end
                % 再次确定顶点位置
                [va, lo] = max(maxv);   % va-顶点的值；lo-顶点在maxv中的位置
                pos = maxl(lo);         % 顶点在功率谱中的位置
                
            end
            % 若pos = -1，即峰值序列已经为空，退出循环
            if pos < 0
                break
            end
            
            % 扩展方位矢量
            Aug_t = [Aug_t, A(:,pos)];
            A(:,pos) = 0;
            Aug_t_s = Aug_t;  % 对于复数矩阵， .' 是普通转置而非共轭转置
            aug_x = pinv(Aug_t_s' * Aug_t_s) * Aug_t_s' * signal;   % pinv：Moore-Penrose 伪逆，这一步是求AX=signal的最小二乘解，X=(A_H*A)^-1*A_H*signal。采用伪逆是防止(A_H * A)为奇异矩阵
            
            % 求残差
            r_n = signal - Aug_t_s * aug_x;
            pos_arrapy = [pos_arrapy, pos];
            times = times + 1;            
            
        end
        
        % 迭代完成后，为空间谱赋值
        powerSpectrumInSpace(pos_arrapy) = abs(aug_x);
        

        
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
