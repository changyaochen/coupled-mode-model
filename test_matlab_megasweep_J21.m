% --- this is to load the time-domain solution calucated from python
% --- and process them as I did for the real experiment data

addpath('/Users/changyaochen/Documents/MATLAB/Matlab library');
% addpath('/Users/jdong/Documents/MATLAB/Matlab library');
close all; clear; clc;
% Q1, Q2, bound = 100000, 100000, 0.0002;
% beta = 0.005;
% gain = 1000.0;
% J21, J12 = 0.0001, 0.001; 


cd('/Users/changyaochen/Box Sync/python_garage/coupled mode model/')
filename = 'test_mega_steady_state_J21.txt';
data = load(filename);

mode_1 = mega2matrix(data(:,[1,2,end]));
mode_1_log = mode_1;
mode_1_log(2:end, 2:end)= log10(mode_1(2:end, 2:end));

colormap jet;
figure(1);
megaPlot(mode_1, '');
prettifyPlot('Frequency (a.u.)','parameter = J21',' Mode 1');
ylim([1.08, 1.12]);
set(gca,'xscale','log');

figure(11);
colormap jet;
megaPlot(mode_1_log, '');
prettifyPlot('Frequency (a.u.)','parameter = J21','Mode 1, log');
ylim([1.08, 1.12]);
set(gca,'xscale','log');

% ============ mode 2 ====================
mode_2 = mega2matrix(data(:,[1,3,end]));
mode_2_log = mode_1;
mode_2_log(2:end, 2:end)= log10(mode_1(2:end, 2:end));

colormap jet;
figure(2);
megaPlot(mode_2, '');
prettifyPlot('Frequency (a.u.)','parameter = J21',' Mode 2');
ylim([1.08, 1.12]);
set(gca,'xscale','log');

figure(21);
colormap jet;
megaPlot(mode_2_log, '');
prettifyPlot('Frequency (a.u.)','parameter = J21',' Mode 2, log');
ylim([1.08, 1.12]);
set(gca,'xscale','log');


% =====================================================================
% ======= to find the sideband (beat frequency) =======================
% =====================================================================

paras = mode_1(1,2:end);
bare_mode_1 = mode_1(2:end, :);
bare_mode_1 = flipud(bare_mode_1);
beat_freq = -1*ones(size(paras));
for i = 2:size(bare_mode_1,2)
    [pks, locs] = findpeaks(bare_mode_1(:,i), bare_mode_1(:,1),'SortStr','descend');
    beat_freq(i-1) = abs(locs(1) - locs(2));
end

figure(90); hold on;
plot(paras, beat_freq,'o', 'markersize', 15);
prettifyPlot('Beat frequency (a.u.)', 'parameter = J21 (a.u.)','');
J12 = 0.001;

