function [outFreqSignal] = musicAlg(signal,L,D)
%MUSICALG ����Ӧmusic�㷨
%   ����1ά�źţ�����thresh�ָ���������������ֵ�����Ƶ�ʷ������
%   signal: 1ά����ʱ���źţ���ʽ��channel��n
%   L��ɨ��Ƕ�����
%   D: Ŀ�����Ŀ
    
    % Ĭ�ϲ�����ֵ
    if ~exist('D','var')
        D = 0;
    end
    
    lamda = 1;
    d=lamda/2;
    [M, objs] = size(signal);
    % filename1 = ['rawsig_',datestr(now,'HHMMSSFFF'),'.mat'];
    % save(filename1,'signal_1d');
    % L = 360;
    % N:���߸���
    N=M;
    %% �����ݲ��м���MUSIC�ı�Ҫ��
    if objs > 0
        % ����correlation matrix�����ڲ����������ޣ�ֻ���ý����źŵ�covariance matrix�������
        % �������������仯��һ�׵��������źſռ�������ռ�
        X = signal*signal';% '���ǹ���ת��
        % filename2 = ['covmat_',datestr(now,'HHMMSSFFF'),'.mat'];
        % save(filename2,'X');
        [eigVec,eigVal] = eig(X);
        eigVal = diag(eigVal);
    %     figure(5);
    %     plot(eigVal,'*');
    %     hold on;
        % ���������ֵ��1/10Ϊ��ֵ��С�ڵľ���Ϊ�����ӿռ��Ӧ������ֵ
        eig_max = max(eigVal);
        eig_min = min(eigVal);
        eig_n_idx = []; % �����ӿռ��Ӧ������ֵ
        if abs(eig_max/eig_min) > 100 && D > 0
            % ����ֵ֮��Ĳ�ֵ����100������Ϊ�����ź��ӿռ䣬����MUSIC
            for i = 1:size(eigVal)
                if eigVal(i) < eig_max * 0.1
                    eig_n_idx = [eig_n_idx, i];
                end
            end
            % �������ø����Ŀ����Ŀ
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
            %% ɨ�����нǶȣ������׷������
            antennaArr = linspace(0,(N-1)*d,N)';
            sine_theta = -2*((-L/2:L/2)/L)/(2*d);
            L_theta = asin(sine_theta);
            % scanAngle = linspace(-pi/2,pi/2,L)';
            scanAngle = L_theta';
            powerSpectrumInSpace = zeros(1,L);
            for i =1:L
                av = array_response_vector(antennaArr,scanAngle(i));
                powerSpectrumInSpace(i) = 1/norm(Qn'*av);
%                 if powerSpectrumInSpace(i) < 1.1% ������ֵ����ʱ����
%                     powerSpectrumInSpace(i) = 1;
%                 end
            end
            % ����ӳ��
    %         figure(5);
    %         plot(powerSpectrumInSpace);
    %         hold on;
    %         powerSpectrumInSpace = (powerSpectrumInSpace-min(powerSpectrumInSpace))/(max(powerSpectrumInSpace)-min(powerSpectrumInSpace));
            % outFreqSignal = powerSpectrumInSpace/max(powerSpectrumInSpace);
    %         powerSpectrumInSpace = powerSpectrumInSpace / min(powerSpectrumInSpace);

        else
            % ������Ϊ�������ź��ӿռ䣬ֻ�������ӿռ�
            powerSpectrumInSpace = ones(1,L);
        end
    else
        % ����ֱ�ӷ��ؾ�����
        powerSpectrumInSpace = ones(1,L);
    end

    outFreqSignal = powerSpectrumInSpace;

end
