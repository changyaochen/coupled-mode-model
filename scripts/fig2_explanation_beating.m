% This file is to read and process the time series data acquired from
% oscilloscope
% if there are multiple .csv to be processed, they should have same
% time-base, i.e., from one measurement!

% Changyao

addpath('/Users/changyaochen/Documents/MATLAB/Matlab library');
addpath('/Users/jdong/Documents/MATLAB/Matlab library');
cd('/Volumes/LaCie/Noise/Data/ARG/S13/3/DC_bias');

clc;
clear;
close all;
format longEng;

% ===== The purpose is to process the "beating" during t_coherent =====
filename_all = {
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_30kHz.', 0, 30, 1;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_32kHz.', 0, 32, 1;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_35kHz.', 0, 35, 1;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_38kHz.', 0, 38, 1;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_40kHz.', 0, 40, 1;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0, 22, 2; 
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.04, 23, 2;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.', 0.05, 24, 2;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_25kHz.', 0.07, 25, 2;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_26kHz.', 0.08, 26, 2;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0, 21, 3;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.06, 22, 3;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.1, 23, 3;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.', 0.12, 24, 3;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p8kHz.', 0.138, 24.8, 3;
% % 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p8kHz_60kHz_50mVpp_off_drive.', 0.13, 24.8, 3;
% % 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p8kHz_60kHz_100mVpp_off_drive.', 0.13, 24.8, 3;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.', 0.0, 20, 4;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0.06, 21, 4;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.11, 22, 4;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.135, 23, 4;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.', 0.16, 24, 4;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p3kHz.', 0.17, 24.3, 4;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.', 0.005, 20, 5;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0.08, 21, 5;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.14, 22, 5;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.18, 23, 5;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p5kHz.', 0.18, 23.5, 5;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.', 0.005, 20, 6;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0.11, 21, 6;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.16, 22, 6;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.18, 23, 6;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p8kHz.', 0.2, 23.8, 6;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0.008, 20, 7;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20p5kHz.',0.0817, 20.5, 7;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.117, 21, 7;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.164, 22, 7;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.',0.19, 23, 7;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p5kHz.',0.2, 23.5, 7;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0.01, 20, 8;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.12, 21, 8;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.16, 22, 8;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p4kHz.',0.2, 23.4, 8;
% 'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0.01, 20,  9;
% 'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.13, 21, 9;
% 'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.16, 22, 9;
'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p3kHz.',0.2, 23.3, 9;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0, 20, 10;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.096, 21, 10;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.15, 22, 10;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p5kHz.',0.18, 23.5, 10;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.',0.2, 24, 10;
    };
time_window = 0;
dataFit     = 0;   % if == 1, then fit the data!
dataRTSA    = 1;   % if == 1, then do the RTSA!
    closeRTSA   = 0;   % if == 1, then close the RTSA figure!
SlopeAll = [];
gammas = [];

trace_num = size(filename_all, 1);
color_map = hsv(trace_num);
leg = [];
timeSlice = 100;
SampleRate = 5e6;  % predefined... later I actually calculate this sample rate from raw data
% f0 = 61.34e3; % for ARG_S13_2_bII
f0 = 61.7e3; % for ARG_S13_3_cIII
fLPF = 60e3; % low corner frequncy for later digital filter
fHPF = 70e3; % high corner frequncy for later digital filter

totalGain    = 0.577 * 0.5 *1e6 * 1e2;  % first factor of 0.577 is accounted for the power splitter (1/sqrt(3)), seconed factor of 0.5 is accounted for the impedance mismatch, 1e6 is FEMTO gain, 1e2 si SIM-911 gain
% totalGain    = 0.707 *1e6 * 1e2;  % first factor of 0.717 is accounted for the power splitter (1/sqrt(2)), 1e6 is FEMTO gain, 1e2 si SIM-911 gain
gap          = 2e-6;  % gap between fingers, in unit of meter
epsilon      = 8.86e-12; % vacuum permittivity, SI unit
omega        = 2*pi*65e3;  % nominal angular vibrational frequency, in unit of rad/s
h            = 10e-6;      % thicknees of the fingers, in unit of meter
N            = 50;         % 2 x number of fingers
VtoA         = @(V, Vdc) V * gap / N / totalGain / Vdc / epsilon / h / omega;
feedback_phases = [];
decay_constant = [];
decay_constant_freq = [];
vac_before_all = [];


for i = 1:trace_num
    if exist([filename_all{i,1},'mat']) == 0 % the .mat doesn't exist
        filename = [filename_all{i,1},'csv'];
        disp('Reading .csv file....');
        tic();
        raw_data = csvread(filename, 10, 0);  % skip the first 10 rows
        toc();
        time = raw_data(:,4);
        eval(strcat('volt1 = raw_data(:,5);'));
        toSave = raw_data(:,[4,5]);
        save([filename_all{i},'mat'],'toSave','-mat');
    else % the .mat file exist
        disp('Reading .mat file....');
        tic();
        S = load([filename_all{i,1},'mat']);
        raw_data = S.toSave;
        time = raw_data(:,1);
        eval(strcat('volt1 = raw_data(:,2);'));
        toc();
    end
    Vdc  = filename_all{i,end}; % DC bias, in V
    fc = 1e3 * filename_all{i,3};    % corner frequency of HPF, in Hz
    
    % ===== calculate feedback phase =====
    [dump, phase_shift_bkg] = Butterworth('low', 8, 150e3, f0);    % the phase shift caused by LPF
    [Mag_temp, phase_temp] = Butterworth('high', 8, fc, f0);    % the phase shift caused by HPF
    feedback_delay = phase_shift_bkg + phase_temp + 90;    % the value 90 is for velocity --> displacement phase shift
    feedback_phases = [feedback_phases; Vdc, fc, feedback_delay];    % add to summary
    
    len  = length(time);
    triggerIdx = find(time > -0.0, 1);
    SampleRate = 1/mean(diff(time));
    
    % ======== processing data, such as filtering =========
    time_before = time(1:triggerIdx - 1); % time before ringdown
    time  = time(triggerIdx:end);
    volt1_before = volt1(1:triggerIdx - 1); % voltage before ringdown portion
    volt1 = volt1(triggerIdx:end);   % only analyze after ringdown
    
    % for some reason, I like to include t < 0 portion
    time = [time_before; time]; volt1 = [volt1_before; volt1];
    
    coherent_idx = find(time > filename_all{i,2}, 1);
    coherent_data = [time(1:coherent_idx), volt1(1:coherent_idx)];
    
    % I will try to bandpass the data between 60 and 70 kHz
    bpFilt = designfilt('bandpassiir','FilterOrder',20, ...
        'HalfPowerFrequency1',fLPF,'HalfPowerFrequency2',fHPF, ...
        'SampleRate',SampleRate);  % the sampling rate is found from the data
    
    voltFiltered = filter(bpFilt, volt1);
%         figure(77);
%         plot(time, voltFiltered, 'k'); 
%         prettifyPlot('Amplitude (V)','Time (s)','');
%         grid off;
%     
    %     figure(10); hold on;
    %     myFFT(time_before, volt1_before,'psd');
    %     xlim([62e3 68e3]);
    
    
    % ======= RTSA processing =========
    if dataRTSA == 1
        figure(100 + 10*Vdc + i);
        [M, Result] = RTSA([time, voltFiltered], timeSlice); colormap jet;
        f0s = Result(:,2);
        f0 = f0s(1);
        % shading interp;
        prettifyPlot('Frequency (Hz)', 'Time (s)',''); grid off;
        ylim([60e3 66.5e3]);
        if closeRTSA == 1
            close();
        end
    end
   
    
    % ===== displacement ringdown processing =====
    displacement = 1e6 * VtoA(voltFiltered, Vdc);   % displacement, in um
    env = abs(hilbert(displacement));
    % fix the amplitude jump at trigger
    ratio = mean(env(triggerIdx:triggerIdx + floor(0.1*(coherent_idx - triggerIdx))))...
           /mean(env(floor(0.9*triggerIdx):triggerIdx));    %  ratio = after / before
    
    vac_before = mean(abs(hilbert(voltFiltered(1:triggerIdx)))) * ratio;    % average vac amplitude, before ringdown, corrected for the ratio
    vac_before_all = [vac_before_all; vac_before/sqrt(2)];
    displacement(1:triggerIdx) = ratio * displacement(1:triggerIdx);
    % recalculate the envelop
    env = abs(hilbert(displacement));
    
    % I am going to low pass this envelope
    lpFilt_fc = 1e3*[1.5];
    for j = 1:size(lpFilt_fc,2)
        lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
            'PassbandFrequency',lpFilt_fc(j),'PassbandRipple',0.2, ...
            'SampleRate',SampleRate);  % the sampling rate is found from the data
        envFiltered = filter(lpFilt, env);
        figure(91); hold on;  % offseted plot
        plot(time(1:1000:coherent_idx), 1.1^(i + j)*envFiltered(1:1000:coherent_idx),'color',color_map(i,:));
        prettifyPlot('Amplitude (um), offseted','time (s)',''); box on;
        %     set(gca, 'yscale', 'log');
        xlim([0 0.2]);
        ylim([1 3]);
        drawnow;
        
        figure(92); hold on;
        plot(time(1:1000:coherent_idx), envFiltered(1:1000:coherent_idx),'color',color_map(i,:));
        prettifyPlot('Amplitude (um)','time (s)',''); box on;
        %     set(gca, 'yscale', 'log');
        xlim([0 0.2]);
        ylim([1 3]);
        drawnow;
        
        % ===== to find the peaks, semi-manually =====
        figure(92);
        %     [pks,locs,widths,proms] = findpeaks(envFiltered(1:1:coherent_idx),time(1:1:coherent_idx), 'MinPeakProminence', 100e-3, 'Annotate', 'extents');
        params = {
            1, 0.8e-3, 10e-3;
            2, 0.8e-3, 10e-3;
            3, 0.8e-3, 10e-3;
            4, 1.2e-3, 5e-3;
            5, 0.8e-3, 10e-3;
            6, 0.8e-3, 10e-3;
            7, 1.0e-3, 5e-3;
            8, 0.9e-3, 6e-3;
            9, 0.8e-3, 10e-3;
            10, 0.8e-3, 25e-3;
            };
        [pks,locs,widths,proms] = findpeaks(envFiltered(1:1:coherent_idx),time(1:1:coherent_idx), 'MinPeakDistance', params{Vdc, 2},'MinPeakProminence', params{Vdc, 3});
%         text(locs,pks,num2str((1:numel(pks))'));
        % to process the found peaks:
        time_idx = -1*ones(size(locs,1)-1, 1);
        beat_freq = time_idx;
        for j = 2:size(locs)
            time_idx(j) = (locs(j) + locs(j-1))/2;
            beat_freq(j) = 1/(locs(j) - locs(j-1));
        end
        figure(93);
        plot(time_idx, beat_freq, 'o');
        prettifyPlot('Beat frequency (Hz)', 'Time (s)','');
        xlim([time(1) filename_all{i, 2}]);
        % to overlay with the RTSA data
        if dataRTSA == 1 && closeRTSA ~= 1
            figure(100 + 10*Vdc + i); hold on;
            plot3(time_idx, Result(1,2) + beat_freq, 1e3*ones(size(time_idx)),'wo','markersize', 10);
        end
    end
        
%     figure(94); hold on;
%     myFFT(time(1:1:coherent_idx), envFiltered(1:1:coherent_idx), 'psd');
%      xlim([0 1e4]);
%     set(gca,'xscale', 'log');
    
end

disp('Finished!')
return;





