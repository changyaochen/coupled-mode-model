% --- this is to load the time-domain solution calucated from python
% --- and process them as I did for the real experiment data

addpath('/Users/changyaochen/Documents/MATLAB/Matlab library');
% addpath('/Users/jdong/Documents/MATLAB/Matlab library');
close all; clear; clc;
% Q1, Q2, bound = 122680.0, 86505.0, 5e-3;
% beta = 5e-3;
% gain = 1000.0;
% J21, J12 = 7.6e-4, 1.0e-3; 
omega1 = 61.4e3;

timeSlice = 100;
ext_copy = 1;
noise_level = 0.1;
cd('/Users/changyaochen/Box Sync/python_garage/coupled mode model/')

% =========== first steady-state ==================
filename = 'solve_single_case_to_fit_data_ss_1_to_3.';
if exist([filename,'mat']) == 0 % the .mat doesn't exist
    filename_h = [filename,'dat'];
    disp('Reading .dat file....');
    tic();
    raw_data_ss = load(filename_h);
    toc();
    toSave = raw_data_ss(:,[1,2]);
    save([filename,'mat'],'toSave','-mat');
else % the .mat file exist
    disp('Reading .mat file....');
    tic();
    S = load([filename,'mat']);
    raw_data_ss = S.toSave;
end
data_ss = raw_data_ss;    % make a copy!
toc();

figure(1); colormap jet;
% ===== convert to 'real' time ======
data_ss(:,1) = raw_data_ss(:,1)/2/pi/omega1;
M_ss = RTSA_v2(data_ss(:,[1,2]), timeSlice, ext_copy, noise_level);
prettifyPlot('Frequency', 'Time', 'steady-state');
ylim([61e3, 66e3])

% % plot the ss FFT
% figure(2);
% plot(M_ss(2:end, 1), log10(M_ss(2:end, end)), 'b');
% prettifyPlot('Power (a.u.)', 'Frequency (Hz)','steady-state');
% xlim([61e3, 66e3])


% =========== then ringdown ==================
filename = 'solve_single_case_to_fit_data_rd_1_to_3.';
if exist([filename,'mat']) == 0 % the .mat doesn't exist
    filename_h = [filename,'dat'];
    disp('Reading .dat file....');
    tic();
    raw_data_rd = load(filename_h);
    toc();
    toSave = raw_data_rd(:,[1,2]);
    save([filename,'mat'],'toSave','-mat');
else % the .mat file exist
    disp('Reading .mat file....');
    tic();
    S = load([filename,'mat']);
    raw_data_rd = S.toSave;
end
data_rd = raw_data_rd;    % make a copy!
toc();

figure(10); colormap jet;
% ===== convert to 'real' time ======
data_rd(:,1) = raw_data_rd(:,1)/2/pi/omega1;
M_rd = RTSA_v2(data_rd(:,[1,2]), timeSlice, ext_copy, noise_level);
prettifyPlot('Frequency', 'Time', 'ringdown');
ylim([61e3, 66e3])


% =========== combine! ==================
temp = data_ss(:,[1,2]);
temp(:,1) = temp(:,1) - temp(end,1);
data_all = [temp; data_rd(:,[1,2])];
M_all = RTSA_v2(data_all(:,[1,2]), 2*timeSlice, ext_copy, noise_level);
prettifyPlot('Frequency', 'Time', 'ringdown');
ylim([61e3, 66e3])

% ========= do the FFT during ss =========
length = int64(0.2*(size(data_ss,1)));
time = data_ss(length:end, 1);
volt = data_ss(length:end, 2);
figure(50);
[f, output, f0] = myFFT(time, volt, 'psd');
prettifyPlot('Power (a.u.)', 'Frequency (Hz)','steady-state');
xlim([64e3, 66e3]); grid off;




