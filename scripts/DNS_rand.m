fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\AB_envelope\'
%% Chance level computation
OT_setup

Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 6;
testfold = 1;
auditory = {'envelope onset'};
N = 1000%random interations
% auditory= {'melmenv','melonsmenv'}
for s=1:length(sbj)
    
    
    for k=1:2
        
        EEG = [];
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        reg = zeros(N,2,EEG.nbchan);
        
        
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
        menv = extract_stimulus2(EEG, PATH, 'envelope', k,sbj{s},task);

        
        %start the random loop
        for r = 1:N
            rand_idx = randperm(length(binIndices));
            
            
            bin_rand = binIndices(rand_idx);
            
            %         binned envelope
            env_bin = zeros(length(binEdges_dB),length(menv));
            
            
            for i = 1:length(binIndices)
                env_bin(bin_rand(i),i) = menv(i);
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
                    mlpt_weight(s,k,r,:,:,:) = model_train.w;
                    
                end
                
                
                %predict the neural data
                [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
                
                reg_int(ao,:) = STATS.r;
            end
            reg(r,:,:) = reg_int;
            reg_int = [];
            sprintf('iteration %d/1000 subject %d',r,s)
        end
        
        result_reg(s,k,:,:,:) = reg;
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
save('DNS_rand.mat','-struct','analysis08_results')  


figure
plot(model_train.t,squeeze(mean(mean(mean(single_weight,2),4),1)))

%data for plotting
temp_dat = squeeze(mean(mean(mean(mean(mlpt_weight,1),2),6),3));

%data for statistical testing
stattemp_dat = squeeze(mean(mean(mlpt_weight,2),6));



bin_plot = binEdges_dB;
bin_plot(sum(temp_dat,2) ==0) = [];
temp_dat(sum(temp_dat,2) ==0,:) = [];
figure;hold on;
for i =1:size(temp_dat,1)
    plot(trf_time,temp_dat(i,:)+(i*10),'linew',2)
end
legend(num2str(bin_plot'))

%find the N1 and P2 peaks for the different bins 
N1 = [30 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

[n1_peak,n1_lat] = min(temp_dat(:,n1_idx(1):n1_idx(2)),[],2)
[p2_peak,p2_lat] = max(temp_dat(:,p2_idx(1):p2_idx(2)),[],2)

%for the statistical testing
n1_peak_rnd = min(stattemp_dat(:,:,n1_idx(1):n1_idx(2)),[],3)
p2_peak_rnd = max(stattemp_dat(:,:,p2_idx(1):p2_idx(2)),[],3)


n1_time = trf_time(n1_idx(1):n1_idx(2));
p2_time = trf_time(p2_idx(1):p2_idx(2));

figure
t = tiledlayout(2,2)
nexttile
for i =1:size(temp_dat,1)
    plot(trf_time,temp_dat(i,:)+(i*10),'Color',[0 0.4470 0.7410],'linew',2)
    hold on
end
xlabel('Time Lag (ms)');
ylabel('Density Bins (dB)');
grid on 
box off

nexttile
imagesc(trf_time,bin_plot,temp_dat)
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
set(gca,'YDir','normal') 
set(gca,'FontSize',16)
box off

nexttile
plot(n1_time(n1_lat),bin_plot(1:end-1), '-o', 'LineWidth', 1.5);
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

nexttile
plot((p2_peak-n1_peak), bin_plot(1:end-1), '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Magnitude a.u.');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([8 14]);
ylim([-0.1 1.1])
% Adding a title or label similar to the "C"
title('N1-P2 peak to peak')
set(gca,'FontSize',16)
box off

title(t,'Chance level DNS TRF','FontSize',28)
set(gca,'FontSize',16)


%% compute the chance level of peak to peak amplitude
for s = 1:length(sbj)
    for n = 1:N
        [n1_peak,n1_lat] = min(squeeze(stattemp_dat(s,n,:,n1_idx(1):n1_idx(2))),[],2);
        [p2_peak,p2_lat] = max(squeeze(stattemp_dat(s,n,:,p2_idx(1):p2_idx(2))),[],2);

        peak_dif(s,n,:) = (p2_peak-n1_peak)';
    end
end

%compute the chance interval per participant
data = squeeze(mean(peak_dif,1));

save('DNS_CI_rand_env_pp.mat','data') 
%% compute the chance level of the N1 amplitude
for s = 1:length(sbj)
    for n = 1:N
        [n1_peak,n1_lat] = min(squeeze(stattemp_dat(s,n,:,n1_idx(1):n1_idx(2))),[],2);
        
        peakn1(s,n,:) = n1_peak';
    end
end

data = squeeze(mean(peak_dif,1));

save('DNS_CIn1_rand_env_pp.mat','data') 

