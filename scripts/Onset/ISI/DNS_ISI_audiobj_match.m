%% Auditory object analysis --> match the %global paths
OT_setup 

DNS_setup

%set the upper limit of onset distances we want to look at -> the bin that
%has been shown to be relevant/ the cutoff
max_value = 100;
min_value = 50;

% Bin the data
binIndices = discretize(dns_dist, binEdges{i});
[counts, ~, binIndices_env] = histcounts(dns_dist, binEdges{i});

ep_t = [-0.1 0.5];


for s=1:length(sbj)
    
    for k=1:2
        
        %% compute the envelopes
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        cd(PATH)
        
        
        novelty_ultm = load(sprintf('ons_ult_%s',task{k}));
        
        fs_new = EEG.srate;
        
        peak = smooth_peak(novelty_ultm.novelty_ultm,fs_new,'sigma',4);
        
        
        %get the indicies
        ons_idx = find(peak);
        
        %find the distance
        ons_dif = diff(ons_idx);
        
        %remove the first one
        ons_idx(1) = [];
        
        %extract the mel Spectrogram
        mel_spec = extract_stimulus2(EEG,PATH,'mel',k,sbj{s},task);
        
        %equalize the data length
        %get the neural data
        resp = double(EEG.data');
        
        if size(resp,1)>size(mel_spec,1)
            resp = resp(1:size(mel_spec,1),:);
        elseif size(resp,1)<size(mel_spec,1)
            mel_spec = mel_spec(1:length(resp),:);
        end
        
        %remove the onsets above the cutoff of, needs to be done via loop
        %--> ons of 3s i need the previous one to compute similarity
        ons_idx_sav = [];
        for on = 2:length(ons_idx)
            if ons_dif(on) < max_value
                %save the index and that of the preceeding ons
                win_len = 5;%ons_dif(on)-2;
                if ons_idx(on)+win_len< length(mel_spec)
                    %get the mel spectrogram of the corresponding segments
                    melspec_pre = normalize(mel_spec(ons_idx(on-1):ons_idx(on-1)+win_len,:),2,'range');
                    melspec_post = normalize(mel_spec(ons_idx(on):ons_idx(on)+win_len,:),2,'range');
                    
                    %get the difference
                    euclidean_distance = norm(melspec_post(:) -melspec_pre(:))/numel(melspec_pre);
                    
                    %save the index and that of the preceeding ons
                    ons_idx_sav = [ons_idx_sav; ons_idx(on-1) ons_idx(on) ons_dif(on) euclidean_distance];
                end
            end
        end
        
        %normalize the values ... why, because too small values are
        %confusing to look at
        %ons_idx_sav(:,4) = normalize(ons_idx_sav(:,4),'range');
        
        %remove the lengthy ones
        ons_idx_sav(find(ons_idx_sav(:,2)+50 > EEG.pnts),:) = [];
        %remove onsets that are closer than 300ms -> overlap 
        ons_idx_sav(find(ons_idx_sav(:,3) < min_value),:) = [];
        
        %create the onset vector based on the proceeding onsets
        onset = zeros(length(EEG.data),1);
        onset(ons_idx_sav(:,2),1) = 1;
        
        %and now extract the corresponding neural model ... 
        %maybe epoch?
        EEG_epo = OT_epochize(EEG,onset,ep_t,0);
        erp_time = linspace(ep_t(1),ep_t(2),size(EEG_epo,2));
        
        EEG.data = EEG_epo;
        EEG.times = erp_time;
        EEG.trials = size(EEG_epo,3);
        EEG.pnts = size(EEG_epo,2);
        %% Baseline Correction 
        base_idx = [ep_t(1)+0.01 -0.01]*EEG.srate;
        b = dsearchn(erp_time',base_idx');
        
        
        %subtract the data 
        EEG_rm = pop_rmbase(EEG,[-0.090 -0.010]);
        
        %save the erp structure
        sav_eeg{s,k} = EEG_rm.data;
        
        %save the euclidean distance
        sav_dist{s,k} = ons_idx_sav;
    end
end

        
fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\auditory_Obj\ '


temp_dist = [];
%get the global distribution 
for s = 1:length(sbj)
   
    temp_ecld = cat(1,sav_dist{s,:});
    temp_dist = [temp_dist; temp_ecld(:,4)];
    
end

figure
hist(temp_dist)

%get the global bin edges
bin2 = quantile(temp_dist, linspace(0, 1, 3 + 1));

%plot the results and sort them according to their euclidean distance
agg_dat=[];
for s = 1:length(sbj)
    
    temp_dat = cat(3,sav_eeg{s,:});
    temp_ecld = cat(1,sav_dist{s,:});
    temp_ecld = temp_ecld(:,4);
    
    
    %sort the data according to ecld
    [temp_ecld_sort ecld_idx] = sort(temp_ecld,'descend');
    temp_dat = squeeze(mean(temp_dat(:,:,ecld_idx),1));
    
    %cleanse excessive epochs
    
    
    %divide into 4 bins
    % uniform distribution
    
%     bin = quantile(temp_ecld, linspace(0, 1, 3 + 1));
   
    binIndices = discretize(temp_ecld_sort, bin2);
    %average over these fuckers
    for bi = 1:length(unique(binIndices))
        agg_dat(s,:,bi) = mean(temp_dat(:,binIndices == bi),2);
    end
end


figure, hold on
for i = 1:size(agg_dat,3)
%     if i == 2; continue; end
    plot_dat = agg_dat(:,:,i);
    plot_dat(12,:) = [];
%     %Compute the mean and SEM
%     meanData = mean(plot_dat,1); % Mean across rows (observations)
%     SEM = std(plot_dat, 0, 1) ./ sqrt(size(plot_dat, 1)); % SEM calculation
%     
%     % Time vector (assume 1 unit per time point)
% %     time = linspace(-100, 500, size(plot_dat, 2)); % Adjust as per your time points
%     
%     % Plot the mean
%     p(i) = plot(erp_time, meanData, 'LineWidth', 2); % Mean line (red)
%     c_lor(i,:) = get(p(i),'Color');
%     % Plot shaded SEM as error bands
%     fill([erp_time, fliplr(erp_time)], ...
%         [meanData + SEM, fliplr(meanData - SEM)], ...
%         c_lor(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'k'); % Shaded area
    figure
    imagesc(plot_dat)
    %
end
legend(p,{'most similar','moderate','very different'},'box','off')

% legend(p,{'most similar','moderate','weakly','very different'},'box','off')
xlabel('Time in m.s.')
ylabel('microvolts')
set(gca,'FontSize',16)

save_fig(gcf,fig_path,'DNS_ISI_audi_obj')


%% test whether there are potential P2 differences

N1 = [0.01 0.150];
P2 = [0.090 0.250];

n1_idx = dsearchn(erp_time',N1');
p2_idx = dsearchn(erp_time',P2');

%contrast N1
n1_dat = squeeze(min(agg_dat(:,n1_idx(1):n1_idx(2),:),[],2));
n1_val = signrank(n1_dat(:,1),n1_dat(:,3));

%contrast P2
p2_dat = max(agg_dat(:,p2_idx(1):p2_idx(2),:),[],2);
p2_val = signrank(p2_dat(:,1),p2_dat(:,3));

%contrast P2-N1 complex
dif_dat = squeeze(max(agg_dat(:,p2_idx(1):p2_idx(2),:),[],2)) - squeeze(min(agg_dat(:,n1_idx(1):n1_idx(2),:),[],2));
dif_val = signrank(dif_dat(:,1),dif_dat(:,3));


figure
violinplot(dif_dat,{'most similar','moderate','weakly','very different'},...
    'ViolinColor',c_lor);

           

%% statistical comparison 
num_participants = 20;
num_conditions = 4;
num_timepoints = size(agg_dat,2);

% Simulated data: (Replace this with your actual data)
data_matrix = agg_dat;

% Convert to FieldTrip structure
data.time  = erp_time; % Assuming time from 0 to 1 sec
data.label = {'Condition1', 'Condition2', 'Condition3', 'Condition4'};
data.dimord = 'subj_chan_time'; % Subject x Channel x Time
for i = 1:num_participants
    data.trial{i} = squeeze(data_matrix(i,:,:)); % Extract participant’s data
    data.subj{i} = ['subj' num2str(i)]; % Subject IDs
end


cfg = [];
cfg.method = 'montecarlo'; % Use permutation-based testing
cfg.statistic = 'depsamplesFmultivariate'; % For repeated measures
cfg.correctm = 'cluster'; % Apply cluster correction
cfg.clusteralpha = 0.05; % Uncorrected cluster threshold
cfg.clusterstatistic = 'maxsum'; % Sum of t-values as cluster statistic
cfg.minnbchan = 0; % Minimum number of neighboring time points for a cluster
cfg.tail = 0; % Two-tailed test
cfg.clustertail = 0;
cfg.alpha = 0.05; % Final corrected p-value threshold
cfg.numrandomization = 1000; % Number of permutations

% Define the design matrix (participants & conditions)
design = zeros(2, num_participants * num_conditions);
design(1,:) = repmat(1:num_participants, 1, num_conditions); % Subject IDs
design(2,:) = reshape(repmat(1:num_conditions, num_participants, 1), 1, []); % Condition labels

cfg.design = design;
cfg.ivar = 2; % Independent variable (condition)
cfg.uvar = 1; % Subject variable

% Run the cluster-based permutation test
stat = ft_timelockstatistics(cfg, data);




            
            
            
            