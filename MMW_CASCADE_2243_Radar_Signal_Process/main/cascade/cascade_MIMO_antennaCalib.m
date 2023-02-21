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

% cascade_MIMO_antennaCalib.m
%  
% Top level main test chain to perform antenna calibration. 
% genCalibrationMatrixCascade module is first initialized before
% actually used in the chain. The output is saved to a .mat file to be used
% later in data processing
%%% MIMO模式的标定程序（应该是）

clearvars
close all


pro_path = getenv('CASCADE_SIGNAL_PROCESSING_CHAIN_MIMO');
input_path = strcat(pro_path,'.\main\cascade\input\');
dataPlatform = 'TDA2'

%%% 数据文件路径。使用正常采集的数据即可，要求空旷场地or暗室，正前方摆放角反，角反距离已知 %%%
% dataFolder_calib_data = '.\main\cascade\testVector\test1\';
% dataFolder_calib_data = 'C:\ti\mmwave_studio_02_01_00_00\mmWaveStudio\PostProc\MIMO_Calibration_Capture\';
% dataFolder_calib_data = 'D:\radardata\cascade\Cascade_PhaseShifterCal_testdata_1112_data\';
% dataFolder_calib_data = 'D:\radardata\cascade\20211227-menkou-jiaofan\Cascade_Capture_22xx_2021.12.27-jiaofan6m\';
% dataFolder_calib_data = 'D:\radardata\cascade\20220329-elevation\Cascade_Capture_22xx_2022.03.29_jiaofan_2\';
dataFolder_calib_data = 'D:\radardata\cascade\20220401-elevation\Cascade_Capture_22xx_2022.04.01_6m_2\';

%%% 目标（角反射器）摆放距离 %%%
targetRange = 6; %target aproximate distance for local maximum search

%%% 参数文件路径（接下来要把参数写进参数文件） %%%
%parameter file name for the test
pathGenParaFile = [input_path,'generateClibrationMatrix_param.m'];

%important to clear the same.m file, since Matlab does not clear cache
%automatically
clear(pathGenParaFile); % 清除文件中原有的参数
%generate parameter file for the test to run
% 生成参数文件，将参数写入到参数文件
parameter_file_gen_antennaCalib_json(dataFolder_calib_data, pathGenParaFile, dataPlatform);

%%% 构造标定对象（初始化） %%%
genCalibrationMatrixObj      = genCalibrationMatrixCascade('pfile', pathGenParaFile,...
    'calibrateFileName',dataFolder_calib_data, 'targetRange', targetRange);

[fileIdx_unique] = getUniqueFileIdx(dataFolder_calib_data); % 在路径dataFolder_test中，读取*_data.bin的编号
[fileNameStruct]= getBinFileNames_withIdx(dataFolder_calib_data, fileIdx_unique{1})	% 获取数据的文件名
genCalibrationMatrixObj.binDataFile = fileNameStruct;% dataFolder_calib_data;%[dataFolder_calib_data listing.name];

if length(genCalibrationMatrixObj.TxToEnable)< 12
    % 确保12个发射天线的数据都在，才可以进行标定
    %it is important to know that all 12 TX needs to be turned on in the MIMO mode to generate a correct calibration matrix. 
    error('This data set cannot be used for calibration, all 12 channels should be enabled');
end

%calibrateValFileNameSave: file name to save calibration results. This file
%will be saved in "dataFolder_calib" after running calibration
% 文件保存路径（最终的标定结果）
calibrateValFileNameSave =[input_path '\calibrateResults_high.mat'];

%use second frame for calibration 
% 使用第2帧进行标定（只要这一帧就行了）
genCalibrationMatrixObj.frameIdx = 2;

%%% 标定 %%%
calibResult = dataPath(genCalibrationMatrixObj);

%%% 标定结果赋值 %%% 
RangeMat = calibResult.RangeMat;
targetRange_est = (floor(mean(RangeMat(:))/genCalibrationMatrixObj.calibrationInterp))...
    *genCalibrationMatrixObj.rangeResolution;
disp(['Target is estimated at range ' num2str(targetRange_est)]);

figure(1);
plot(RangeMat(:));grid on;
title('peak index across all channels')

figure(2);
plot(squeeze(abs(calibResult.Rx_fft(:,1,:))))
%just to make it compatible with old data
params.Slope_MHzperus = genCalibrationMatrixObj.Slope_calib/1e12;
params.Sampling_Rate_sps = genCalibrationMatrixObj.Sampling_Rate_sps;
%save the calibration data
save(calibrateValFileNameSave, 'calibResult','params');
