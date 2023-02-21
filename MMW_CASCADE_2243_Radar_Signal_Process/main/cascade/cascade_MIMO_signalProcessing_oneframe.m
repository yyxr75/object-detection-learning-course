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

% cascade_MIMO_signalProcessing_oneframe.m
%
% Top level main test chain to process the raw ADC data. The processing
% chain including adc data calibration module, range FFT module, DopplerFFT
% module, CFAR module, DOA module. Each module is first initialized before
% actually used in the chain.

% 修改后的signalProcessing主程序，用于读取单帧的data文件
% 当前处于master分支下，用于普通的MIMO模式，仅调整了一些chirp参数保证测距范围

% 输出图表：figure 1: 2*2图，分别为range、range-doppler、x-y heat-map、3D points
%          figure 2: 1*2图，分别为静态、动态的heat-map
%          figure 3: 1*1图，静态的range-angle图
%          figure 4: 1*2图，由3D解算数据在垂直方向叠起得到的Range Angle图和xy图

clearvars
close all

PLOT_ON = 1; % 1: turn plot on; 0: turn plot off
LOG_ON = 1; % 1: log10 scale; 0: linear scale
% numFrames_toRun = 10; %number of frame to run, can be less than the frame saved in the raw data
SAVEOUTPUT_ON = 0;
PARAM_FILE_GEN_ON = 1;
DISPLAY_RANGE_AZIMUTH_DYNAMIC_HEATMAP = 1 ; % Will make things slower % 是否显示动态/静态RangeAngle热力图的对比图，默认为0，这个不影响数据处理，只跟显示有关
dataPlatform = 'TDA2'
DYNAMIC = 1;        % 采用何种方法计算动态目标角度 0-FFT，1-MUSIC 刘明旭 20220217
STATIC = 1;         % 采用何种方法计算静态场景角度 0-FFT，1-CBF，2-Capon，3-Compressed Sensing OMP，4-Compressed Sensing L1 Optimization，5-APES，6-Backward Forward APES，7 IAA-APES 刘明旭 20220217

%% get the input path and testList
% 读取文件路径
pro_path = getenv('CASCADE_SIGNAL_PROCESSING_CHAIN_MIMO');
% input_path = strcat(pro_path,'\main\cascade\input\');
input_path = '.\main\cascade\input\';
testList = strcat(input_path,'testList_oneframe.txtt');
%path for input folder
fidList = fopen(testList,'r');
testID = 1;




while ~feof(fidList)
    
    %% get each test vectors within the test list
    % 按照路径列表，分别分析数据
    % test data file name 
    % 例程的数据文件路径（json文件格式。在同一路径下，只能有一个*.mmwave.json，该json存储部分参数，以chirp相关的参数为主
    dataFolder_test = fgetl(fidList);    
   
    % calibration file name
    dataFolder_calib = fgetl(fidList);
    
    % module_param_file defines parameters to init each signal processing module
    % 用于初始化每个信号处理模块的参数文件，该.m文件存储部分参数
    module_param_file = fgetl(fidList);
    
    % parameter file name for the test
    % 例程的参数文件test1_param.m
    pathGenParaFile = [input_path,'test',num2str(testID), '_param.m'];
    %important to clear the same.m file, since Matlab does not clear cache
    %automatically
    clear(pathGenParaFile);
    
    % generate parameter file for the test to run 
    % 读取.json文件中的chirp参数，并为运行的例程生成参数文件test1_param.m
    if PARAM_FILE_GEN_ON == 1     
        parameter_file_gen_json(dataFolder_test, dataFolder_calib, module_param_file, pathGenParaFile, dataPlatform);
    end
    
    %load calibration parameters
    load(dataFolder_calib)
    
    % simTopObj is used for top level parameter parsing and data loading and saving
    % 从生成的参数文件test1_param.m中读取参数，存入变量中
    % 这里是以面向对象的形式写的，每一部分的处理都是一个类
    % 都是Module的子类
    simTopObj           = simTopCascade('pfile', pathGenParaFile);
    calibrationObj      = calibrationCascade('pfile', pathGenParaFile, 'calibrationfilePath', dataFolder_calib);
    rangeFFTObj         = rangeProcCascade('pfile', pathGenParaFile);
    DopplerFFTObj       = DopplerProcClutterRemove('pfile', pathGenParaFile);
    detectionObj        = CFAR_CASO('pfile', pathGenParaFile);
    DOAObj              = DOACascade('pfile', pathGenParaFile);
    
    % get system level variables
    % 获取系统信息
    platform            = simTopObj.platform;       % 系统平台（级联板为'TI_4Chip_CASCADE'）
    numValidFrames      = simTopObj.totNumFrames;   % 有效帧数
    cnt = 1;
    frameCountGlobal = 0;                           % 已读取的总帧数
    
    
	% Get Unique File Idxs in the "dataFolder_test" 
	% 读取路径下，*data.bin文件的编号
	[fileIdx_unique] = getUniqueFileIdx(dataFolder_test);
    
    % 按照编号逐一读取（每一帧一个编号）
	for i_file = 1:(length(fileIdx_unique))
        
        % Get File Names for the Master, Slave1, Slave2, Slave3   
        % 根据编号，获取文件名，只有_data.bin，不需要再搞Idx
        [fileNameStruct]= getBinFileNames_noIdx(dataFolder_test, fileIdx_unique{i_file});        
        %pass the Data File to the calibration Object
        calibrationObj.binfilePath = fileNameStruct;

        detection_results = [];              
            
        % 从这里开始，逐帧解析
        % 计时
        tic
        %read and calibrate raw ADC data
        % 读取数据，校正
        % 数据维度：[numSamplePerChirp, numLoops, numRX, numChirpPerLoop]
        frameIdx = str2num(fileIdx_unique{i_file});
        calibrationObj.frameIdx = 1;    % 由于每个文件只有1帧，因此这里始终置为1
        frameCountGlobal = frameCountGlobal + 1
        adcData = datapath(calibrationObj);% 类calibrationCascade的类函数datapath(obj)。读取数据，按照所需的维数排列；并根据标定参数进行校正

        % RX Channel re-ordering
        % 根据RX顺序，调整数据numRX中的顺序
        adcData = adcData(:,:,calibrationObj.RxForMIMOProcess,:);            

        %only take TX and RXs required for MIMO data analysis
        % adcData = adcData

        if mod(frameIdx, 10)==1
            fprintf('Processing %3d frame...\n', frameIdx);
        end


        %perform 2D FFT
        rangeFFTOut = [];
        DopplerFFTOut = [];

        % adcData维度: [numSamplePerChirp, numLoops, numRX, numChirpPerLoop]
        % 其中numChirpPerLoop == numTX
        for i_tx = 1: size(adcData,4)
            %% 按照不同TX分别处理，对每组数据分别进行进行rangeFFT和DopplerFFT
            % range FFT
            rangeFFTOut(:,:,:,i_tx)     = datapath(rangeFFTObj, adcData(:,:,:,i_tx));       % 类rangeProcCascade的类函数datapath(obj, input)
            % Doppler FFT
            DopplerFFTOut(:,:,:,i_tx)   = datapath(DopplerFFTObj, rangeFFTOut(:,:,:,i_tx)); % 类DopplerProcClutterRemove的类函数datapath(obj, input)
        end            

        % CFAR done along only TX and RX used in MIMO array
        % 重构Doppler输出：
        % [numSamplePerChirp, numLoops, numRX, numChirpPerLoop]转为[numSamplePerChirp, numLoops, numRx*numTx]
        DopplerFFTOut = reshape(DopplerFFTOut,size(DopplerFFTOut,1), size(DopplerFFTOut,2), size(DopplerFFTOut,3)*size(DopplerFFTOut,4));

        %detection
        % 各个天线的数据取平方后进行累积，对累积结果取10log10
        % 输出维度为[numSamplePerChirp, numLoops]
        sig_integrate = 10*log10(sum((abs(DopplerFFTOut)).^2,3) + 1);

        detection_results = datapath(detectionObj, DopplerFFTOut);  % 类CFAR_CASO的类函数datapath(obj, input)
        detection_results_all{cnt} =  detection_results;

        % 检测到的目标点
        detect_all_points = [];
        for iobj = 1:length(detection_results)
            detect_all_points (iobj,1)=detection_results(iobj).rangeInd+1;
            detect_all_points (iobj,2)=detection_results(iobj).dopplerInd_org+1;
            detect_all_points (iobj,4)=detection_results(iobj).estSNR;
        end

        % 出图（range图和range-doppler图）
        if PLOT_ON
            %% 出图（range图和range-doppler图）
            figure(1);
            set(gcf,'units','normalized','outerposition',[0 0 1 1])

            %%% range图    Range Profile(zero Doppler - thick green line)
            subplot(2,2,1)
            plot((1:size(sig_integrate,1))*detectionObj.rangeBinSize, sig_integrate(:,size(sig_integrate,2)/2+1),'g','LineWidth',4);hold on; grid on % 绘制doppler=0的range频谱（那条绿线）
            for ii=1:size(sig_integrate,2)
                %% 逐一绘制每个doppler bin的频谱
                plot((1:size(sig_integrate,1))*detectionObj.rangeBinSize, sig_integrate(:,ii));hold on; grid on
                if ~isempty(detection_results)
                    ind = find(detect_all_points(:,2)==ii);
                    if (~isempty(ind))
                        rangeInd = detect_all_points(ind,1);
                        plot(rangeInd*detectionObj.rangeBinSize, sig_integrate(rangeInd,ii),'o','LineWidth',2,...
                            'MarkerEdgeColor','k',...
                            'MarkerFaceColor',[.49 1 .63],...
                            'MarkerSize',6);
                    end
                end
            end

            %title(['FrameID: ' num2str(cnt)]);
            xlabel('Range(m)');
            ylabel('Receive Power (dB)')
            title(['Range Profile(zero Doppler - thick green line): frameID ' num2str(frameIdx)]);
            hold off;

            %%% range-doppler图    Range/Velocity Plot
            subplot(2,2,2);
            %subplot_tight(2,2,2,0.1)
            imagesc((sig_integrate))
            % 在range-velocity图中显示CFAR结果。刘明旭 20211125
            hold on;
            for iobj = 1:size(detect_all_points, 1)
                plot(detect_all_points(iobj, 2),detect_all_points(iobj, 1), 'o','LineWidth',2);
                hold on;
            end
            hold off;

            c = colorbar;
            c.Label.String = 'Relative Power(dB)';
            title(' Range/Velocity Plot');
            pause(0.01)

            %%%
%                 figure(2);
%                 for i=1:size(sig_integrate, 1)
%                     plot((1:size(sig_integrate, 2))*detectionObj.velocityBinSize, sig_integrate(i,:));
%                     hold on; 
%                 end
            %%%
        end

        % 角度估计
        angles_all_points = [];
        xyz = [];
        %if 0
        if ~isempty(detection_results)
            %% 只有在range/ddoppler CFAR中成功检测到目标点，才会进入角度估计
            % DOA, the results include detection results + angle estimation results.
            % access data with angleEst{frame}(objectIdx).fieldName
            angleEst = datapath(DOAObj, detection_results); % 类DOACascade的类函数datapath(obj, detected_obj)

            if length(angleEst) > 0
                %% 根据输出结果逐一赋值，准备出图
                for iobj = 1:length(angleEst)
                    angles_all_points (iobj,1:2)=angleEst(iobj).angles(1:2);
                    angles_all_points (iobj,3)=angleEst(iobj).estSNR;
                    angles_all_points (iobj,4)=angleEst(iobj).rangeInd;
                    angles_all_points (iobj,5)=angleEst(iobj).doppler_corr;
                    angles_all_points (iobj,6)=angleEst(iobj).range;
                    %switch left and right, the azimuth angle is flipped
                    xyz(iobj,1) = angles_all_points (iobj,6)*sind(angles_all_points (iobj,1)*-1)*cosd(angles_all_points (iobj,2));
                    xyz(iobj,2) = angles_all_points (iobj,6)*cosd(angles_all_points (iobj,1)*-1)*cosd(angles_all_points (iobj,2));
                    %switch upside and down, the elevation angle is flipped
                    xyz(iobj,3) = angles_all_points (iobj,6)*sind(angles_all_points (iobj,2)*-1);
                    xyz(iobj,4) = angleEst(iobj).doppler_corr;
                    xyz(iobj,9) = angleEst(iobj).dopplerInd_org;
                    xyz(iobj,5) = angleEst(iobj).range;
                    xyz(iobj,6) = angleEst(iobj).estSNR;
                    xyz(iobj,7) = angleEst(iobj).doppler_corr_overlap;
                    xyz(iobj,8) = angleEst(iobj).doppler_corr_FFT;

                end
                angles_all_all{cnt} = angles_all_points;
                xyz_all{cnt}  = xyz;
                maxRangeShow = detectionObj.rangeBinSize*rangeFFTObj.rangeFFTSize;
                %tic
                if PLOT_ON
                    %% 出图
                    moveID = find(abs(xyz(:,4))>=0);
                    subplot(2,2,4);  % 绘制CFAR所选的3D点                      

                    if cnt==1
                        scatter3(xyz(moveID,1),xyz(moveID,2),xyz(moveID,3),45,(xyz(moveID,4)),'filled');
                    else
                        yz = [xyz_all{cnt}; xyz_all{cnt-1}];
                        scatter3(xyz(moveID,1),xyz(moveID,2),xyz(moveID,3),45,(xyz(moveID,4)),'filled');
                    end

                    c = colorbar;
                    c.Label.String = 'velocity (m/s)';                        
                    grid on;

                    xlim([-20 20])
                    ylim([1 maxRangeShow])
                    %zlim([-4 4])
                    zlim([-5 5])
                    xlabel('X (m)')
                    ylabel('y (m)')
                    zlabel('Z (m)')                        

%                         view([-9 15])
                    view([0 0 1])
                    title(' 3D point cloud');

                    %plot range and azimuth heatmap
                    % Range-Angle热力图，在这里还要再算一次角度FFT
                    subplot(2,2,3)
                    STATIC_ONLY = 1;    % 是否只显示静止目标的Range-Angle Map
                    minRangeBinKeep =  5;
                    rightRangeBinDiscard =  20;
                    [mag_data_static(:,:,frameCountGlobal) mag_data_dynamic(:,:,frameCountGlobal) y_axis x_axis]= plot_range_azimuth_2D(frameIdx, detectionObj.rangeBinSize, DopplerFFTOut,...
                        length(calibrationObj.IdTxForMIMOProcess),length(calibrationObj.RxForMIMOProcess), ...
                        detectionObj.antenna_azimuthonly, LOG_ON, STATIC_ONLY, PLOT_ON, minRangeBinKeep, rightRangeBinDiscard,detection_results, DYNAMIC, STATIC);



                if (DISPLAY_RANGE_AZIMUTH_DYNAMIC_HEATMAP) % 是否显示动态/静态RangeAngle热力图的对比图，默认为0，这个不影响数据处理，只跟显示有关
                    %% 显示对比图
                    figure(2)
                    subplot(121);
                    surf(y_axis, x_axis, (mag_data_static(:,:,frameCountGlobal)).^0.4,'EdgeColor','none');
                    view(2);
                    xlabel('meters');    ylabel('meters')
                    title({'Static Range-Azimuth Heatmap',strcat('Current Frame Number = ', num2str(frameCountGlobal))})

                    subplot(122);
                    surf(y_axis, x_axis, (mag_data_dynamic(:,:,frameCountGlobal)).^0.4,'EdgeColor','none');
                    view(2);    
                    xlabel('meters');    ylabel('meters')
                    title('Dynamic HeatMap')
                end
                pause(0.1)


                end

            end

        end
        %% 在range-FFT和doppler-FFT的基础上，做3D-DOA
        % TODO: 在此之前可以搞个2D CFAR，针对动态部分，之前的默认的CFAR太容易漏检了。
        % TODO: 3D DOA 函数应该在这里插进去。不受之前的range、doppler维度上的CFAR干扰，直接做DOA（天线参数：DOAObj.D）
        angles_DOA_az = [-70, 0.5, 70];
        angles_DOA_ele = [-20, 0.5, 20];
%         plot_range_azimuth_3D(frameIdx, detectionObj.rangeBinSize, DopplerFFTOut,...
%                         length(calibrationObj.IdTxForMIMOProcess),length(calibrationObj.RxForMIMOProcess), ...
%                         DOAObj.D, LOG_ON, PLOT_ON, minRangeBinKeep, rightRangeBinDiscard, angles_DOA_az, angles_DOA_ele, detection_results, DYNAMIC, STATIC);

        %% 当前帧的所有处理结束
        cnt = cnt + 1;    
        toc    
        
        
    end
    
    ind = strfind(dataFolder_test, '\');
    testName = dataFolder_test(ind(end-1)+1:(ind(end)-1));
    if SAVEOUTPUT_ON == 1
        save(['.\main\cascade\output\newOutput_',testName,'.mat'],'angles_all_all', 'detection_results_all','xyz_all');
    end
    testID = testID + 1;
    
end
