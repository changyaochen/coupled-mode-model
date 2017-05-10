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

filename_all = {
% % 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p8kHz_60kHz_50mVpp_off_drive.', 0.13, 24.8, 3;
% % 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p8kHz_60kHz_100mVpp_off_drive.', 0.13, 24.8, 3;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_30kHz.', 0, 30, 1, 64800.28;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_32kHz.', 0, 32, 1, 64844.7326;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_35kHz.', 0, 35, 1, 64844.7326;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_38kHz.', 0, 38, 1, 64844.7326;
% 'B123_ARG_S13_3_ringdown_Vg=1V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_40kHz.', 0, 40, 1, 64800.28;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0, 22, 2,  64711.3987;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.04, 23, 2, 64889.1772;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.', 0.05, 24, 2, 64889.1772;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_25kHz.', 0.07, 25, 2, 64889.1772;
% 'B123_ARG_S13_3_ringdown_Vg=2V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_26kHz.', 0.08, 26, 2, 64889.1772;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0, 21, 3, 64800.28;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.06, 22, 3, 64889.1772;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.1, 23, 3, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.', 0.12, 24, 3, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=3V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p8kHz.', 0.138, 24.8, 3, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.', 0.0, 20, 4, 64178.0630;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0.06, 21, 4, 64889.1772;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.11, 22, 4, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.135, 23, 4, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.', 0.16, 24, 4, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=4V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24p3kHz.', 0.17, 24.3, 4, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.', 0.0, 20, 5, 64444.7308;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0.08, 21, 5, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.14, 22, 5, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.18, 23, 5, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=5V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p5kHz.', 0.18, 23.5, 5, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.', 0, 20, 6, 64666.9540;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.', 0.11, 21, 6, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.', 0.16, 22, 6, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.', 0.18, 23, 6, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=6V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p8kHz.', 0.2, 23.8, 6, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0.008, 20, 7, 64889.1772;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20p5kHz.',0.0817, 20.5, 7, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.117, 21, 7, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.164, 22, 7, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23kHz.',0.19, 23, 7, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=7V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p5kHz.',0.2, 23.5, 7, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0.01, 20, 8, 64666.9540;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.12, 21, 8, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.16, 22, 8, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=8V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p4kHz.',0.2, 23.4, 8, 64933.6219;
'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0.01, 20,  9, 64889.1772;
'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.13, 21, 9, 64933.6219;
'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.16, 22, 9, 64933.6219;
'B123_ARG_S13_3_ringdown_Vg=9V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p3kHz.',0.2, 23.3, 9, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_20kHz.',0.01, 20, 10, 64444.7308;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_21kHz.',0.096, 21, 10, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_22kHz.',0.15, 22, 10, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_23p5kHz.',0.18, 23.5, 10, 64933.6219;
% 'B123_ARG_S13_3_ringdown_Vg=10V_FEMTO_1uAV_SIM911_100x_LPF_150kHz_HPF_24kHz.',0.2, 24, 10, 64933.621;
    };
time_window = 0;
dataFit     = 1;   % if == 1, then fit the data!
dataRTSA    = 0;   % if == 1, then do the RTSA!
showRaw     = 1;   % if == 1, then show the raw oscillation, with env 
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
    Vdc  = filename_all{i,4}; % DC bias, in V
    fc = 1e3 * filename_all{i,3};    % corner frequency of HPF, in Hz
    f_init = filename_all{i,5};    % initial frequency of the ringdown, in Hz
    
    % ===== calculate feedback phase =====
    [dump, phase_shift_bkg] = Butterworth('low', 8, 150e3, f_init);    % the phase shift caused by LPF
    [Mag_temp, phase_temp] = Butterworth('high', 8, fc, f_init);    % the phase shift caused by HPF
    feedback_delay = phase_shift_bkg + phase_temp + 90;    % the value 90 is for velocity --> displacement phase shift
    feedback_phases = [feedback_phases; Vdc, fc, feedback_delay];    % add to summary
    
    len  = length(time);
    triggerIdx = find(time > -0.0, 1);
    SampleRate = 1/mean(diff(time));
    
    % ======== processing data, such as filtering =========
    time_before = time(1:triggerIdx); % time before ringdown
    time  = time(triggerIdx:end);
    coherent_idx = find(time > filename_all{i,2}, 1);
    coherent_data = [time(1:coherent_idx), volt1(1:coherent_idx)];
    idx0 = find(time > 0.002, 1);
    volt1_before = volt1(1:triggerIdx); % voltage before ringdown portion
    volt1 = volt1(triggerIdx:end);   % only analyze after ringdown
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
        ylim([60e3 65e3]);
%         close();
        
        %  ========== Now I wil try to fit the f vs t with exponential decay
        if dataFit == 1
            toFit = Result(Result(:,1) > filename_all{i,2}, 1:2);
            toFit = toFit(toFit(:,1) < 1.5, :);
            F_freq = @(p,x) p(1) * exp(- x/p(2)) + 60.76e3; % p(2) should be time constant
            opts = optimset('Algorithm', 'levenberg-marquardt','TolFun',1e-7,'TolX',1e-7);
            p0 = [5e3, 1];
            [p1,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(F_freq,p0, toFit(:,1),toFit(:,2),[],[],opts);
                J = jacobian;
                r = residual;
                [Q,R] = qr(J,0);
                mse = sum(abs(r).^2)/(size(J,1)-size(J,2));
                Rinv = inv(R);
                Sigma = Rinv*Rinv'*mse;
                se = sqrt(diag(Sigma));
                se = se';
            disp(['Frequency decay time constant for frequency is ', num2str(p1(2)), ' s']);
            decay_constant_freq = [decay_constant_freq; Vdc, feedback_delay, p1(2), se(2)];
            tPlot = linspace(filename_all{i,2}, Result(end,1), 100);
            freqFit = F_freq(p1, tPlot);
                   % ===== to plot the fitting separately =====
                   figure(99); hold on;
                   plot(toFit(:,1), toFit(:,2), 'o');
                   plot(tPlot, freqFit, 'r-');
                   prettifyPlot('Frequency (Hz)', 'Time (s)','');
                   drawnow;
        end

    end
   
    
    % ===== displacement ringdown processing =====
    displacement = 1e6 * VtoA(voltFiltered, Vdc);   % displacement, in um
    temp = [time, displacement];
    toPlot = temp(1:floor(len/len):end,:);
    % plot(toFit(:,1), toFit(:,2),'bo');
    env = abs(hilbert(displacement));
    % I am going to low pass this envelope
    lpFilt = designfilt('lowpassiir','FilterOrder',2, ...
        'PassbandFrequency',1e3,'PassbandRipple',0.2, ...
        'SampleRate',SampleRate);  % the sampling rate is found from the data
    envFiltered = filter(lpFilt, env);
    
    figure(92); hold on;
    plot(time(1:1000:end), 1.1^(i) * envFiltered(1:1000:end),'color',color_map(i,:));
    prettifyPlot('Amplitude (a.u.)','time (s)',''); box on;
    
    F1 = @(p,x) p(1) * exp(- x/p(2)) + p(3); % p(2) should be time constant
    
    % ringdown with nonlinear damping, following Damian's notation,
    % y is velocity, x is time
    % p(1) is q, p(2) is eta/f_0^2, p(3) is the initial velocity
    F3 = @(p,x) exp(-x/2/p(1)) ./ sqrt(1/p(3)/p(3) + p(2)*(1 - exp(-x/p(1))));
    opts = optimset('Algorithm', 'levenberg-marquardt','TolFun',1e-7,'TolX',1e-7);
    
    if dataFit == 1
        toFit = [time(1:1000:end), envFiltered(1:1000:end)];
        toFit = toFit(toFit(:,1)>filename_all{i,2}, :);  % after t_coherent
        toFit = toFit(toFit(:,1)<2, :);    % before 2 seconds
       
        % ===== now toFit consists of [time, amplitude] =====
        p0 = [2, .1, .1];
        [p1,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(F1,p0, toFit(:,1),toFit(:,2),[],[],opts);
            J = jacobian;
            r = residual;
            [Q,R] = qr(J,0);
            mse = sum(abs(r).^2)/(size(J,1)-size(J,2));
            Rinv = inv(R);
            Sigma = Rinv*Rinv'*mse;
            se = sqrt(diag(Sigma));
            se = se';
        gammas = [gammas; 1/p1(2)/pi];
        tPlot = time(coherent_idx:1e3:end);
        figure(92); hold on;
        plot(tPlot,  1.1^(i) * F1(p1, tPlot), 'k-');
        disp(['Decay time constant for amplitude is ', num2str(p1(2)), ' s']);
        decay_constant = [decay_constant; Vdc, feedback_delay, p1(2), se(2)];
    end
    
    leg = feedback_phases(:,3);
    if dataFit == 1
        temp = ones(2*size(leg,1), 1);
        temp(1:2:end) = leg;
        temp(2:2:end) = leg;
        leg = temp;
    end
    legend(num2str(leg));
    set(gca, 'yscale', 'log');
    xlim([0 2]);
    ylim([.2e-1 2]);
    drawnow;
    
    if showRaw == 1
       figure( 100 + 10*Vdc + i); hold on;
       plot(time, displacement, 'r-');
       prettifyPlot('Displacement (V)', 'Time (s)', '');
       plot(time(1:10:end), env(1:10:end),'k-', 'linewidth', 3);
    end
    
end

disp('Finished!')
return;





