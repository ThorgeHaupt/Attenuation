fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\AB_envelope\'
%% Model Comparison using variance paritioning
OT_setup

Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 6;
testfold = 1;
auditory = {'envelope onset'};
% auditory= {'melmenv','melonsmenv'}
for s=1:length(sbj)
    
    
    for k=1:2
        
        EEG = [];
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        reg = zeros(length(auditory),EEG.nbchan);
        raw = zeros(length(auditory),EEG.nbchan);
        
        
        %extract the stimulus
        menv = extract_stimulus2(EEG, PATH, auditory{1,1}, k,sbj{s},task);
        
        cd(PATH)
        if k == 1
            
            [audioIn,fs] =audioread(sprintf('narrow_audio_game_%s.wav',sbj{s}));
        else
            
            [audioIn,fs] =audioread(sprintf('wide_audio_game_%s.wav',sbj{s}));
            
        end
        
        [log_mel_spec freq_centers] = log_mel_spectrogram(audioIn,fs,10,25,[64 8000],64);
        
        log_tempdat = log_mel_spec;
        log_tempdat(log_tempdat < 0) = 0;
        log_deriv = diff(log_tempdat,2,2);
        menv = normalize(abs(sum(log_deriv)),'range')';
        
        %% get the rate of change
%         menv = extract_stimulus2(EEG, PATH, 'envelope onset', k,sbj{s},task);
        
        %measure of density using the variance
        win_l = 10*EEG.srate
        
        %zeropad the signal
%         menv_pad = [zeros(1,win_l/2) menv' zeros(1,win_l/2)]';
        
        
        dnsM = movmean(menv,win_l);
        
        %normalize
        dnsM_norm = normalize(dnsM,'range');
        
        
%         %bin those bitches
%         % Define the bin edges in dB
%         binEdges_dB = 30:6:80;  % Binning in 8 dB steps up to 64 dB (adjust as needed)
%         nBins = length(binEdges_dB);
%         
%         % Convert dB edges to linear scale by taking 10^(binEdges_dB/20)
%         binEdges_linear = 10.^(binEdges_dB / 20);
%         
%         % Normalize the bin edges to be between 0 and 1
%         binEdges_normalized = (binEdges_linear - min(binEdges_linear)) / ...
%             (max(binEdges_linear) - min(binEdges_linear));
        
        % Calculate the histogram counts and bin indices using histcounts
        [counts, binEdges_dB, binIndices] = histcounts(dnsM_norm,4);
%         figure
%         histogram('BinEdges',binEdges_dB,'BinCounts',counts)
%         set(gca,'view',[90 -90])
        
        randperm(length(binIndices));
        
        
        
        %         binned envelope
        env_bin = zeros(length(binEdges_dB),length(menv));
        menv = extract_stimulus2(EEG, PATH, 'envelope', k,sbj{s},task);
        
        
        for i = 1:length(binIndices)
            env_bin(binIndices(i),i) = menv(i);
        end
        
        %normalize each bin
        env_bin_norm = normalize(env_bin,2,'range');
        
        %exclude 0 entries
        env_bin_norm(sum(env_bin_norm,2) == 0,:) = [];
        
        stim_col = {menv,env_bin_norm'};
        
        for ao = 1:length(stim_col)
            stim = stim_col{1,ao};
            
            %get the neural data
            resp = double(EEG.data');
            
            if size(resp,1)>size(stim,1)
                resp = resp(1:size(stim,1),:);
            end
            
            [strain,rtrain,stest,rtest] = mTRFpartition(stim,resp,nfold,testfold);
            strainz = strain; %normalization occurs at the stim extraction fun
            stestz = stest;   %normalization occurs at the stim extraction fun
            
            
            rtrainz = cellfun(@(x) zscore(x,0,'all'),rtrain,'UniformOutput',false);
            rtestz = zscore(rtest,[],'all');
            
            %% use cross-validation
            fs = EEG.srate;
            
            
            cv = mTRFcrossval(strainz,rtrainz,fs,Dir,tmin,tmax,lambdas,'Verbose',0);
            
            %get the optimal regression parameter
            l = mean(cv.r,3); %over channels
            [l_val,l_idx] = max(mean(l,1));
            l_opt = lambdas(l_idx);
            
            %save the lambda
            %             l_save(s,k,ao,:) = lambda_opt;
            
            %train the neural model on the optimal regularization parameter
            model_train = mTRFtrain(strainz,rtrainz,fs,Dir,tmin,tmax,l_opt,'verbose',0);
            %                 weights(s,k,ao,:) = squeeze(mean(model_train.w,3));
            if size(model_train.w,1)<2
                single_weight(s,k,:,:) = squeeze(model_train.w);
            else
                mlpt_weight(s,k,:,:,:) = model_train.w;
                
            end
            
            
            %predict the neural data
            [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
            
            reg(ao,:) = STATS.r;
        end
        
        
        result_reg(s,k,:,:) = reg;
        %         result_raw(s,k,:,:) = raw;
        
    end
end

trf_time = model_train.t;
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results')
analysis08_results = struct;
analysis08_results.result_reg = result_reg;
analysis08_results.auditory = auditory;
analysis08_results.single_weight = single_weight;
analysis08_results.mlpt_weight = mlpt_weight;
analysis08_results.trf_time = trf_time;
analysis08_results.descp = 'sorted according to DNS';
save('DNS_ons.mat','-struct','analysis08_results')  
        
% plot the results
temp_dat = squeeze(mean(mean(result_reg,2),4));
Env = temp_dat(:,1);
AB_Env = temp_dat(:,2);
figure;
hold on;

plot(Env, '-o', 'Color', [0, 0.45, 0.74], 'MarkerFaceColor', 'auto', 'DisplayName', 'Env');
plot(AB_Env, '-o', 'Color', [0.49, 0.18, 0.56], 'MarkerFaceColor', 'auto', 'DisplayName', 'AB Env');
% Customize the plot
xlabel('Subjects');
ylabel('Pearson''s r');
legend('Location', 'NorthWest');
title('Plot B');
grid on;

temp_dat = squeeze(mean(mean(result_reg,2),1));

for i = 1:size(temp_dat,1)
    figure
    topoplot(temp_dat(i,:),EEG.chanlocs)
end


%% plot the weights
figure
plot(trf_time,squeeze(mean(mean(mean(single_weight,2),4),1)))

temp_dat = squeeze(mean(mean(mlpt_weight,2),5));
bin_plot = binEdges_dB;
bad_idx = sum(temp_dat,3) ==0
bin_plot(:,bad_idx(1,:)) = [];
temp_dat(:,bad_idx(1,:),:) = [];


%find the N1 and P2 peaks for the different bins 
N1 = [30 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

[n1_peak,n1_lat] = min(temp_dat(:,:,n1_idx(1):n1_idx(2)),[],3);
[p2_peak,p2_lat] = max(temp_dat(:,:,p2_idx(1):p2_idx(2)),[],3);

n1_lat = round(mean(n1_lat));
p2_lat = round(mean(p2_lat));

n1_peak = mean(n1_peak);
p2_peak = mean(p2_peak);

n1_time = trf_time(n1_idx(1):n1_idx(2));
p2_time = trf_time(p2_idx(1):p2_idx(2));


%% plot the min and max values
figure
tiledlayout(2,2)
nexttile
plot(n1_time(n1_lat), bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('DNS Bins (dB)');
grid on;
xlim([-50 200]);
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('N1 Peak latency')

nexttile
plot(p2_time(p2_lat), bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([-50 200]);
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('P2 Peak latency')

nexttile
plot(n1_peak, bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Peak Amplitude');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([-22 1]);
ylim([-0.1 1.1])
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('N1 Peak amplitude')

nexttile
plot(p2_peak, bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Peak Amplitude');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([1 22]);
ylim([-0.1 1.1])
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('P2 Peak amplitude')


%% replicate the exact figure
plot_dat = squeeze(mean(temp_dat,1));
figure
t = tiledlayout(2,2)
nexttile
for i =1:size(temp_dat,2)
    plot(trf_time,plot_dat(i,:)+(i*10),'Color',[0 0.4470 0.7410],'linew',2)
    hold on
end
xlabel('Time Lag (ms)');
ylabel('Density Bins (dB)');
grid on 
set(gca,'FontSize',16)
box off

nexttile
imagesc(trf_time,bin_plot,plot_dat)
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
set(gca,'YDir','normal') 
set(gca,'FontSize',16)
box off

nexttile
plot(n1_time(n1_lat), bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
grid on;
ylim([-0.1 1.1])
xlim([0 400])
% Adding a title or label similar to the "C"
title('N1 Peak latency')
set(gca,'FontSize',16)
box off

load('DNS_CI_rand_env_pp.mat');
%get the 97%interval going
l_bound = prctile(data,2.5);
u_bound = prctile(data,97.5);
m_dat = mean(data);
nexttile
plot((p2_peak-n1_peak), bin_plot, '-o', 'LineWidth', 1.5);
hold on
errorbar(m_dat, bin_plot, (l_bound'-m_dat'), 'horizontal', '-o', 'MarkerSize', 6, 'LineWidth', 1.5, 'CapSize', 10);

% Labeling the axes
xlabel('Magnitude a.u.');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([15 45]);
ylim([-0.1 1.1])
% Adding a title or label similar to the "C"
title('N1-P2 peak to peak')
set(gca,'FontSize',16)
box off

title(t,'DNS based TRF','FontSize',28)
set(gca,'FontSize',16)

% save_fig(gcf,fig_path,'DNS_env')

%% 
temp_dat = flip(temp_dat,1);
figure;hold on;
tiledlayout(size(temp_dat,1),1)
for i =1:size(temp_dat,1)
    nexttile
    plot(model_train.t,temp_dat(i,:))
    axis off
    box off
end

%% per participant
temp_dat = squeeze(mean(mean(mlpt_weight,2),5));
bin_plot = binEdges_dB;
bad_idx = sum(temp_dat,3) ==0
bin_plot(:,bad_idx(1,:)) = [];
temp_dat(:,bad_idx(1,:),:) = [];


%find the N1 and P2 peaks for the different bins 
N1 = [30 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

[n1_peak,n1_lat] = squeeze(mean(min(temp_dat(:,:,n1_idx(1):n1_idx(2)),[],3)));
[p2_peak,p2_lat] = squeeze(mean(max(temp_dat(:,:,p2_idx(1):p2_idx(2)),[],3)));

n1_time = trf_time(n1_idx(1):n1_idx(2));
p2_time = trf_time(p2_idx(1):p2_idx(2));


