%  Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/ 
%  
%  
%   Redistribution and use in source and binary forms, with or without 
%   modification, are permitted provided that the following conditions 
%   are met:
%  
%     Redistributions of source code must retain the above copyright 
%     notice, this list of conditions and the following disclaimer.
%  
%     Redistributions in binary form must reproduce the above copyright
%     notice, this list of conditions and the following disclaimer in the 
%     documentation and/or other materials provided with the   
%     distribution.
%  
%     Neither the name of Texas Instruments Incorporated nor the names of
%     its contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%  
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
%   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
%   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
%   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
%   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
%   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
%   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%  

%CFAR_CASO_Doppler_overlap.m
%
% This function performs 1D CFAR_CASO detection along the
% Doppler direction, and declare detection only if the index overlap with
% range detection results. 

%input
%   obj: object instance of CFAR_CASO
%   Ind_obj_Rag: index of range bins that has been determined by the first
%   step detection along the range direction
%   sigCpml: a 3D complex matrix, range x Doppler x antenna array
%   sig_integ: a 2D real valued matrix, range x Doppler 
% 输入
%   obj: CFAR_CASO对象
%   Ind_obj_Rag: rangeCFAR提取的目标点（下标）
%   sigCpml: 完成RangeFFT和DopplerFFT的信号，维度为[numSamplePerChirp, numLoops, numRx*numTx]
%   sig_integ: 各个天线的数据取平方后进行累积,累积后的维度为[numSamplePerChirp, numLoops]

%output
%   N_obj: number of objects detected
%   Ind_obj: 2D bin index of the detected object
%   noise_obj_an: antenna specific noise estimation before integration


function [N_obj, Ind_obj, noise_obj_an] = CFAR_CASO_Doppler_overlap(obj, Ind_obj_Rag, sigCpml, sig_integ)
    %% 计算dopplerCFAR，采用SO-CFAR
    % 只有当dopplerCFAR结果的下标与rangeCFAR结果下标一致时，才作为结果输出

    % 载入参数
    maxEnable = obj.maxEnable;
    cellNum0 = obj.refWinSize;  % 参考单元长度（单侧），二维向量，(1)是rangeCFAR所用，(2)是dopplerCFAR所用
    gapNum0 = obj.guardWinSize; % 保护单元长度（单侧），二维向量，(1)是rangeCFAR所用，(2)是dopplerCFAR所用
    cellNum = cellNum0(2);      % dopplerCFAR的参考单元长度（单侧）
    gapNum = gapNum0(2);        % dopplerCFAR的保护单元长度（单侧）
    K0 = obj.K0(2);             % dopplerCFAR的阈值

    rangeNumBins = size(sig_integ,1);   % numSamplePerChirp，即数据在range这一维度上的长度

    %extract the detected points after range detection
    % 取出所有rangeCFAR中得到的目标点
    detected_Rag_Cell = unique(Ind_obj_Rag(:,1));   % 取出目标点的下标列表（去重）
    sig = sig_integ(detected_Rag_Cell,:);           % 把这部分数据从sig_integ中提取出来，sig的维度从[numSamplePerChirp, numLoops]变为[numDetectedSample, numLoops]

    M_samp=size(sig, 1);    % numDetectedSample,rangeCFAR得到的目标数
    N_pul=size(sig, 2);     % numChirp，每个天线发射的chirp数，也是numLoops


    %for each point under test, gapNum samples on the two sides are excluded
    %from averaging. Left cellNum/2 and right cellNum/2 samples are used for
    %averaging
    % 对于每一个被测点，两侧gaptot~gapNum这段数据用于平均（长度为cellNum）
    gaptot=gapNum + cellNum;

    % 初始化结果数据
    N_obj=0;
    Ind_obj=[];
    noise_obj_an = [];
    vec=zeros(1,N_pul+gaptot*2);
    
    for k=1:M_samp
        %% 对每个rangeCFAR目标点，在doppler维度上进行CFRA
        
        %get the range index at current range index
        % 获取当前rangeCFAR目标点的下标
        detected_Rag_Cell_i = detected_Rag_Cell(k);             % 获取rangeCFAR目标点的下标（range维度）
        ind1 = find(Ind_obj_Rag(:,1) == detected_Rag_Cell_i);   % 获取rangeCFAR目标点在Ind_obj_Rag中的下标（range维度）
        indR = Ind_obj_Rag(ind1, 2);                            % 获取rangeCFAR目标点在doppler维度上的下标
        %extend the left the vector by copying the left most the right most
        %gaptot samples are not detected.
        % 取出当前doppler维度向量，并对其进行重组（复制最左侧gaptot长度的数据到两端）
        sigv=(sig(k,:));
        vec(1:gaptot) = sigv(end-gaptot+1:end);
        vec(gaptot+1: N_pul+gaptot) = sigv;
        vec(N_pul+gaptot+1:end) = sigv(1:gaptot);
        
        %start to process
        % 初始化当前doppler维度的结果向量，开始计算CFAR
        ind_loc_all = [];
        ind_loc_Dop = [];
        ind_obj_0 = 0;
        noiseEst = zeros(1,N_pul);
        
        for j=1+gaptot:N_pul+gaptot
            %% 对每个点，计算参考单元内的数据总和
            cellInd=[j-gaptot: j-gapNum-1 j+gapNum+1:j+gaptot];
            noiseEst(j-gaptot) = sum(vec(cellInd));
        end
        
        for j=1+gaptot:N_pul+gaptot
            %% 对每个点，计算SO-CFAR
            % 分别两侧参考单元内的数据下标
            j0 = j - gaptot;
            cellInd=[j-gaptot: j-gapNum-1 j+gapNum+1:j+gaptot];
            cellInda = [j-gaptot: j-gapNum-1 ];
            cellIndb =[j+gapNum+1:j+gaptot];
            % 求均值，取min
            cellave1a =sum(vec(cellInda))/(cellNum);
            cellave1b =sum(vec(cellIndb))/(cellNum);
            cellave1 = min(cellave1a,cellave1b);        

            % 判断是否作为dopplerCFAR目标点
            maxInCell = max(vec(cellInd));
            if maxEnable==1
                %% 若标志位为1，判断当前值是否为局部最大值（-gaptot:gaptot之内），在满足门限的同时，需要满足局部最大值才能作为目标点
                %detect only if it is the maximum within window
                condition = ((vec(j)>K0*cellave1)) && ((vec(j)>maxInCell));
            else
                %% 否则，只判断当前值与计算得出的CFAR自适应门限的大小，大于门限则作为目标点
                condition = vec(j)>K0*cellave1;
            end

            if condition==1
                %% 在dopplerCFAR判断为目标点后，判断是否与rangeCFAR重叠
                %check if this detection overlap with the Doppler detection
                if(find(indR == j0))
                    %% 若二者重叠，则加入到结果列表中
                    %find overlap, declare a detection
                    ind_win = detected_Rag_Cell_i;
                    %range index
                    ind_loc_all = [ind_loc_all ind_win];
                    %Doppler index
                    ind_loc_Dop = [ind_loc_Dop j0];
                end

            end

        end
        
        % 整理当前doppler维度的结果
        ind_obj_0 = [];
        if (length(ind_loc_all)>0)
            % 整理当前doppler维度的结果
            ind_obj_0(:,1) = ((ind_loc_all));
            ind_obj_0(:,2) = ind_loc_Dop;
            
            % 加入到全部dopplerCFAR结果中
            if size(Ind_obj,1) == 0
                Ind_obj = ind_obj_0;
            else
                %following process is to avoid replicated detection points
                % 避免目标点重复，在确定没有相同的目标点后，加入全部dopplerCFAR结果中
                ind_obj_0_sum = ind_loc_all + 10000*ind_loc_Dop;
                Ind_obj_sum = Ind_obj(:,1) + 10000*Ind_obj(:,2);
                for ii= 1: length(ind_loc_all)
                    if (length(find(Ind_obj_sum == ind_obj_0_sum(ii)))==0)
                        Ind_obj = [Ind_obj ; ind_obj_0(ii,:)];
                    end
                end
            end
        end

    end

    % 目标点数目
    N_obj = size(Ind_obj,1);

    % 筛选有效目标点，并计算噪声方差
    % 将参数（参考单元数目和保护单元数目）置为range方向的参数
    % reset the ref window size to range direction
    cellNum = cellNum0(1);
    gapNum = gapNum0(1);
    gaptot=gapNum + cellNum;
    % 初始化有效目标点
    % get the noise variance for each antenna
    N_obj_valid = 0;
    Ind_obj_valid = [];
    
    % 筛选有效目标点，并计算噪声方差
    for i_obj = 1:N_obj    
        ind_range = Ind_obj(i_obj,1);
        ind_Dop = Ind_obj(i_obj,2);
        
        % 筛选有效目标点（若目标点处的能量小于阈值（默认是0），则跳过）
        %skip detected points with signal power less than obj.powerThre
        if (min(abs(sigCpml(ind_range, ind_Dop,:)).^2) < obj.powerThre)
            continue;
        end
        % range维度上的参考单元下标
        if ind_range<= gaptot
            %on the left boundary, use the right side samples twice
            cellInd=[ind_range+gapNum+1:ind_range+gaptot ind_range+gapNum+1:ind_range+gaptot];
        elseif ind_range>=rangeNumBins-gaptot+1
            %on the right boundary, use the left side samples twice
            cellInd=[ind_range-gaptot: ind_range-gapNum-1 ind_range-gaptot: ind_range-gapNum-1];
        else
            cellInd=[ind_range-gaptot: ind_range-gapNum-1 ind_range+gapNum+1:ind_range+gaptot];
        end

        % 根据参考单元计算噪声；目标点加入到有效目标点中
        N_obj_valid = N_obj_valid +1;
        noise_obj_an(:, i_obj) = reshape((mean(abs(sigCpml(cellInd, ind_Dop, :)).^2, 1)), obj.numAntenna, 1, 1);
        Ind_obj_valid(N_obj_valid,:) = Ind_obj(i_obj,:);    

    end

    N_obj = N_obj_valid;
    Ind_obj = Ind_obj_valid;






