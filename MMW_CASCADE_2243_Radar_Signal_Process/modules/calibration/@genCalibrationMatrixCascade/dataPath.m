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


%dataPath.m
%
% dataPath function of genCalibrationMatrixCascade module. This function generates the
% calibration maxtrix for frequency and phase calibration. The calibration
% data is given by calibrateFileName. The calibration results are saved in
% calibResult.
%
%input
%   obj: object instance of genCalibrationMatrixCascade



function calibResult = dataPath(obj)
%% 具体的标定过程

%%% 参数 %%% 
Samples_per_Chirp = obj.numSamplePerChirp;
nchirp_loops = obj.nchirp_loops;
Chirps_per_Frame = obj.numChirpsPerFrame;
Sampling_Rate_sps = obj.Sampling_Rate_sps;
TxToEnable = obj.TxToEnable;
numTX = length(obj.TxToEnable);
interp_fact = obj.calibrationInterp;    % rangeFFT时，每个bin的插值数目
Slope_calib = obj.Slope_calib;
frameIdx = obj.frameIdx;
binDataFile = obj.binDataFile;
calibrateFileName = obj.calibrateFileName;
targetRange = obj.targetRange;
RxOrder = obj.RxOrder;

numRX = obj.numRxToEnable;
NumDevices = obj.NumDevices;

%%% 计算角反放置距离对应的range-bin（有一个上限和下限，在这中间寻找精确的值）
% 公式：
%       c * Δf       c * Sampling_Rate_sps * bin_idx                        2 * d * Slope * Samples_per_Chirp
% d = ----------- = -----------------------------------  ====>>>  bin_idx = ------------------------------------
%     2 * Slope       2 * Slope * Samples_per_Chirp                                c * Sampling_Rate_sps
range_bin_search_min = round((Samples_per_Chirp)*interp_fact*((targetRange-2)*2*Slope_calib/(3e8*Sampling_Rate_sps))+1,0);
range_bin_search_max = round((Samples_per_Chirp)*interp_fact*((targetRange+2)*2*Slope_calib/(3e8*Sampling_Rate_sps))+1,0);

switch obj.dataPlatform  
    %% 检查硬件平台是否为TDA2
     case 'TDA2'
        numChirpPerLoop = length(obj.TxToEnable);
        numLoops = obj.nchirp_loops; % Only valid for TDM-MIMO
        numRXPerDevice = 4; % Fixed number
        %%% 读取MIMO模式下的AD数据，并整合到1个数组中。
        %%% 返回的数组维度为 [numSamplePerChirp, numLoops, numRX, numChirpPerLoop]
        [radar_data_Rxchain] = read_ADC_bin_TDA2_separateFiles(obj.binDataFile,frameIdx,Samples_per_Chirp,numChirpPerLoop,numLoops, numRXPerDevice, 1);
    otherwise
        error('Not supported data capture platform!');
end

% Re-ordering of the TX data so that [Master1][Slave1][Slave2][Slave3]
% 将数据按照[Master1][Slave1][Slave2][Slave3]的顺序重新排列
radar_data_Rxchain = radar_data_Rxchain(:,:,:,TxToEnable);

%find the peak location and complex value that corresponds to the calibration target.
% 在每个chirp中寻找角反对应的峰，并记录其幅值、相位
for iTX = 1: numTX
    for iRx = 1:numRX
        Rx_Data = mean(radar_data_Rxchain(:,:,iRx,iTX),2);  % Average chirps within a frame
                                                            % 计算所在虚拟通道，每个采样点的均值（对所在通道内所有loop做平均）
                                                            % 采用均值做range-FFT
        % 计算range-FFT，并提取峰值
        [Rx_fft(:,iRx,iTX),Angle,Val_peak, Rangebin] = ...
            radar_fft_find_peak(obj, Rx_Data,range_bin_search_min, range_bin_search_max);
        AngleMat(iTX,iRx)=Angle;        % 峰值对应的角度
        RangeMat(iTX,iRx)=Rangebin;     % 峰值对应的bin
        PeakValMat(iTX,iRx)=Val_peak;   % 峰值对应的幅值
       
    end
end

%get average phase across RXs for TXs, the average phase will be used for TX phase calibration in TX beamforming, to be added. 
RxMismatch = [];
TxMismatch = [];
%TX index used for tx calibration
TX_ind_calib = TxToEnable(1);

%%% 计算每个接收天线对应的相位均值
in=AngleMat;
Num_Rxs = length(in(1,:));
temp=in(:,1);
temp=in-temp(:,ones(1,Num_Rxs));
for i=1:Num_Rxs
    RxMismatch(i)=Average_Ph(temp(:,i)*pi/180)*180/pi;  % 计算该列的均值（每列是12维，对应1个接收天线的所有虚拟阵元（1接收，对12个发射
end

%%% 计算每个发射天线的相位均值
Num_Txs = length(in(:,1));
temp=in(TX_ind_calib,:);
temp=in-temp(ones(1,Num_Txs),:);
for i=1:Num_Txs
    TxMismatch(i)=Average_Ph(temp(i,:)*pi/180)*180/pi;
end
TxMismatch = reshape(TxMismatch, 3, 4); % 重置尺寸（这又是干啥？

calibResult.AngleMat = AngleMat;
calibResult.RangeMat = RangeMat;
calibResult.PeakValMat = PeakValMat;
calibResult.RxMismatch = RxMismatch;
calibResult.TxMismatch = TxMismatch; 
calibResult.Rx_fft = Rx_fft;


