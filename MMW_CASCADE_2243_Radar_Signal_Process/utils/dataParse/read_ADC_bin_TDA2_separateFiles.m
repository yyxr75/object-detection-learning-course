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

%% read raw adc data with MIMO 

function [radar_data_Rxchain] = read_ADC_bin_TDA2_separateFiles(fileNameCascade,frameIdx,numSamplePerChirp,numChirpPerLoop,numLoops, numRXPerDevice, numDevices)
%% 读取MIMO模式下的AD数据，并整合到1个数组中
    % 返回的数组维度为 [numSamplePerChirp, numLoops, numRX, numChirpPerLoop]

    % 获取文件名
    dataFolder =fileNameCascade.dataFolderName;
    fileFullPath_master = fullfile(dataFolder,fileNameCascade.master);
    fileFullPath_slave1 = fullfile(dataFolder,fileNameCascade.slave1);
    fileFullPath_slave2 = fullfile(dataFolder,fileNameCascade.slave2);
    fileFullPath_slave3 = fullfile(dataFolder,fileNameCascade.slave3);

    % 读取数据的二进制bin文件
    % 读取的数组维度为[numSamplePerChirp, numLoops, numRXPerDevice, numChirpPerLoop]
    [radar_data_Rxchain_master] = readBinFile(fileFullPath_master, frameIdx,numSamplePerChirp,numChirpPerLoop,numLoops, numRXPerDevice, numDevices);
    [radar_data_Rxchain_slave1] = readBinFile(fileFullPath_slave1, frameIdx,numSamplePerChirp,numChirpPerLoop,numLoops, numRXPerDevice, numDevices);
    [radar_data_Rxchain_slave2] = readBinFile(fileFullPath_slave2, frameIdx,numSamplePerChirp,numChirpPerLoop,numLoops, numRXPerDevice, numDevices);
    [radar_data_Rxchain_slave3] = readBinFile(fileFullPath_slave3, frameIdx,numSamplePerChirp,numChirpPerLoop,numLoops, numRXPerDevice, numDevices);

    % Arranged based on Master RxChannels, Slave1 RxChannels, slave2 RxChannels, slave3 RxChannels 
    % The RX channels are re-ordered according to "TI_Cascade_RX_ID" defined in
    % "module_params.m"
    % 在numRXPerDevice这一维度，将4个数组拼接
    % 拼接后的数组维度为[numSamplePerChirp, numLoops, numRX, numChirpPerLoop]
    radar_data_Rxchain(:,:,1:4,:) = radar_data_Rxchain_master;
    radar_data_Rxchain(:,:,5:8,:) = radar_data_Rxchain_slave1;
    radar_data_Rxchain(:,:,9:12,:) = radar_data_Rxchain_slave2;
    radar_data_Rxchain(:,:,13:16,:) = radar_data_Rxchain_slave3;
        
end


function [adcData1Complex] = readBinFile(fileFullPath, frameIdx,numSamplePerChirp,numChirpPerLoop,numLoops, numRXPerDevice, numDevices)
%% 读取bin文件，并转换为复数形式，按照顺序重新排列
    % 返回的数组维度为[numSamplePerChirp, numLoops, numRXPerDevice, numChirpPerLoop]
    
    %% 计算所需AD数据在bin文件中的位置，并载入这部分数据
    Expected_Num_SamplesPerFrame = numSamplePerChirp*numChirpPerLoop*numLoops*numRXPerDevice*2;
    fp = fopen(fileFullPath, 'r');
    fseek(fp,(frameIdx-1)*Expected_Num_SamplesPerFrame*2, 'bof');   % 根据计算好的偏移量，移动文件位置指示符（指针）
    adcData1 = fread(fp,Expected_Num_SamplesPerFrame,'uint16');     % 读取所需部分的AD数据
    neg             = logical(bitget(adcData1, 16));                % 取data的第16位（0 or 1），并转为逻辑值。数据类型为uint16，16位就是最高有效位，即符号位，为1则为负数
    adcData1(neg)    = adcData1(neg) - 2^16;                        % 逻辑值为1的部分，即符号位为负数的，减去2^16，即从补码形式中读取负数
                                                                    % 这里其实和直接以int16的形式读取是一样的，为啥要自己折腾一次？
    
    %% 将数据转为复数形式，按照顺序重新排列
    adcData1 = adcData1(1:2:end) + sqrt(-1)*adcData1(2:2:end);% 排列顺序：实部，虚部
    adcData1Complex = reshape(adcData1, numRXPerDevice, numSamplePerChirp, numChirpPerLoop, numLoops);  % 数组重构
                                                                                                        % 将N*1向量重构为[N_rx, sample, chirp_per_loop, loops]
                                                                                                        % N_rx是单个2243的接收天线数
                                                                                                        % sample是每个chirp的采样点数
                                                                                                        % 这里的chirp_per_loop其实就是发射天线数，TDP模式下，每个loop内每个天线循环发射一次，n个天线就有n个chirp
                                                                                                        % loops就是每个天线发射的chirp数
    adcData1Complex = permute(adcData1Complex, [2 4 1 3]);% 按[2 4 3 1]的顺序置换维度，将数组调整为[numSamplePerChirp, numLoops, numRXPerDevice, numChirpPerLoop]
    fclose(fp);
    
    end