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
%radar_fft_find_peak.m
%
%This function finds the peak location and complex value that corresponds to the
%calibration target.
%
%input
%   obj: object instance of genCalibrationMatrixCascade
%   Rx_Data: input adc data 
%   range_bin_search_min: start of the range bin to search for peak
%   range_bin_search_max: end of the range bin to search for peak
%

% Output:
%   Angle_FFT_Peak: Phase at bin specified by FRI_fixed OR Phase at
%                         highest peak after neglecting DC peaks
%   Val_FFT_Peak: Complex value at bin specified by FRI_fixed OR
%                   Complex value at highest peak after neglecting DC peaks
%   Fund_range_Index: Bin number for highest peak after neglecting DC
%                     peaks
%   Rx_fft: Complex FFT values

function [Rx_fft,Angle_FFT_Peak, Val_FFT_Peak,Fund_range_Index] = radar_fft_find_peak(obj, Rx_Data,range_bin_search_min,range_bin_search_max)
%% 计算当前通道的range-FFT，并提取峰值，以及对应的幅值、相位
% Rx_Data:              对应通道的输入数据，已经过loop的平均
% range_bin_search_min: 寻找峰值的最小bin
% range_bin_search_max: 寻找峰值的最大bin

Effective_Num_Samples = length(Rx_Data);    % 每个Chirp的采样长度
wind = hann_local(Effective_Num_Samples);   % hanning窗函数
wind = wind/rms(wind);                      % 除以窗函数的均方根（这是为啥？）
interp_fact = obj.calibrationInterp;        % rangeFFT时，每个bin的插值数目

%%% range FFT %%%
Rx_Data_prefft = Rx_Data.*wind; % 加窗
Rx_fft = (fft(Rx_Data_prefft,interp_fact*Effective_Num_Samples));   % range FFT

%%% 查找峰值对应的bin %%%
Rx_fft_searchwindow = abs(Rx_fft(range_bin_search_min:range_bin_search_max));
[~,Fund_range_Index]= max(Rx_fft_searchwindow(:));
Fund_range_Index = Fund_range_Index+range_bin_search_min-1;

%%% 峰值对应的角度和幅值 %%%
Angle_FFT_Peak = angle(Rx_fft(Fund_range_Index))*180/pi;
Val_FFT_Peak = Rx_fft(Fund_range_Index);

end