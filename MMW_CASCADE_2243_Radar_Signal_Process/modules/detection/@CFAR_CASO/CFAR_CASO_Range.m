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
% 

%CFAR_CASO_Range.m
%
%This function performs CFAR_CASO detection along range direction

%input
%   obj: object instance of CFAR_CASO
%   sig: a 2D real valued matrix, [range x Doppler]
% 输入：
%   obj: 对象，CFAR_CASO
%   sig: 经过rangeFFT和dopplerFFT，再将不同天线的信号累加后的实信号矩阵，维度为[range x Doppler]

%output
%   N_obj: number of objects detected
%   Ind_obj: 2D bin index of the detected object
%   noise_obj: noise value at the detected point after integration


function [N_obj, Ind_obj, noise_obj, CFAR_SNR] = CFAR_CASO_Range(obj, sig)
    %% 计算rangeCFAR，采用SO-CFAR

    cellNum = obj.refWinSize;   % 参考单元长度（单侧），二维向量，(1)是rangeCFAR所用，(2)是dopplerCFAR所用
    gapNum = obj.guardWinSize;  % 保护单元长度（单侧），二维向量，(1)是rangeCFAR所用，(2)是dopplerCFAR所用
    cellNum = cellNum(1);       % rangeCFAR的参考单元长度（单侧）
    gapNum = gapNum(1);         % rangeCFAR的保护单元长度（单侧）
    K0 = obj.K0(1);             % rangeCFAR的阈值

    M_samp=size(sig, 1);    % 每个chirp的采样点数
    N_pul=size(sig, 2);     % 每个天线的chirp数
    


    %for each point under test, gapNum samples on the two sides are excluded
    %from averaging. Left cellNum/2 and right cellNum/2 samples are used for
    %averaging
    % 对于每一个被测点，两侧gaptot~gapNum这段数据用于平均（长度为cellNum）
    gaptot=gapNum + cellNum;
    % 结果数据初始化
    N_obj=0;
    Ind_obj=[];
    noise_obj = [];
    CFAR_SNR = [];
    
    % 两侧忽略长度
    discardCellLeft = obj.discardCellLeft;
    discardCellRight = obj.discardCellRight;


    %for the first gaptot samples only use the right sample
    % 对每个chirp的信号分别进行CFAR
    for k=1:N_pul        
        % 取出当前chirp向量，并对当前chirp进行重组
        sigv=(sig(:,k))';
        vec = sigv((discardCellLeft+1):(M_samp-discardCellRight));% 去掉两侧忽略的部分
        vecLeft = vec(1:(gaptot));          % 取左侧参考部分（这是干啥？）
        vecRight = vec(end-(gaptot)+1:end); % 取右侧参考部分（这是干啥？）
        vec = [vecLeft vec vecRight];       % 重组vec
        
        % 在有效范围内，逐一对每个点进行CFAR计算
        for j=1:(M_samp-discardCellLeft-discardCellRight)
            % 取参考单元的下标
            cellInd=[j-gaptot: j-gapNum-1 j+gapNum+1:j+gaptot];
            cellInd=cellInd + gaptot;
            cellInda=[j-gaptot: j-gapNum-1];
            cellInda=cellInda + gaptot;
            cellIndb=[ j+gapNum+1:j+gaptot];
            cellIndb=cellIndb + gaptot;
            
            % 计算均值（采用SO-CFAR的方式，取两侧均值的最小值）
            cellave1a =sum(vec(cellInda))/(cellNum);
            cellave1b =sum(vec(cellIndb))/(cellNum);
            cellave1 = min(cellave1a,cellave1b);

            %if((j > discardCellLeft) && (j < (M_samp-discardCellRight)))
            if obj.maxEnable==1 %check if it is local maximum peak
                %% 若标志位为1，判断当前值是否为局部最大值（-gaptot:gaptot之内），需要满足局部最大值才能作为目标点输出
                maxInCell = max(vec([cellInd(1):cellInd(end)]));
                if (vec(j+gaptot)>K0*cellave1 && ( vec(j+gaptot)>=maxInCell))
                    N_obj=N_obj+1;
                    Ind_obj(N_obj,:)=[j+discardCellLeft, k];% 目标点的下标（range和doppler维的位置）
                    noise_obj(N_obj) = cellave1; %save the noise level
                    CFAR_SNR(N_obj) = vec(j+gaptot)/cellave1;
                end
            else
                %% 否则，只判断当前值与计算得出的CFAR自适应门限的大小，大于门限则作为目标点输出
                if vec(j+gaptot)>K0*cellave1
                    N_obj=N_obj+1;
                    Ind_obj(N_obj,:)=[j+discardCellLeft, k];% 目标点的下标（range和doppler维的位置）
                    noise_obj(N_obj) = cellave1; %save the noise level
                    CFAR_SNR(N_obj) = vec(j+gaptot)/cellave1;
                end
            end        
        end
    end

    % get the noise variance for each antenna
    % 获取每个天线的噪声方差
    % 但实际上噪声已经在提取目标点的时候计算过了
    for i_obj = 1:N_obj

        ind_range = Ind_obj(i_obj,1);
        ind_Dop = Ind_obj(i_obj,2);
        if ind_range<= gaptot
            %on the left boundary, use the right side samples twice
            cellInd=[ind_range+gapNum+1:ind_range+gaptot ind_range+gapNum+1:ind_range+gaptot];
        elseif ind_range>=M_samp-gaptot+1
            %on the right boundary, use the left side samples twice
            cellInd=[ind_range-gaptot: ind_range-gapNum-1 ind_range-gaptot: ind_range-gapNum-1];
        else
            cellInd=[ind_range-gaptot: ind_range-gapNum-1 ind_range+gapNum+1:ind_range+gaptot];

        end


    end

