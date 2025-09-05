%% alternative auditory object analysis
OT_setup 

DNS_setup

%set the upper limit of onset distances we want to look at -> the bin that
%has been shown to be relevant/ the cutoff
max_value = 100;
min_value = 20;

% Bin the data
binIndices = discretize(dns_dist, binEdges{i});
[counts, ~, binIndices_env] = histcounts(dns_dist, binEdges{i});

ep_t = [-0.1 0.5];

r_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\eeglab2021\plugins\ICLabel\matconvnet\matlab\compatibility\parallel\';
rmpath(r_path)
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
        env = extract_stimulus2(EEG,PATH,'mTRF envelope',k,sbj{s},task);

%         %extract the mfcc and the pitch
%         [audio, fs] = audioread(sprintf('%s_audio_game_%s.wav',task{k},sbj{s}));
%         mfccs = mfcc(audio,fs);
        
%         save(sprintf('mfcc_%s.mat',task{k}),'mfccs')
%         [f0, ~] = pitch(audio, fs);
%         save(sprintf('pitch_%s.mat',task{k}),'f0')


        %load the two data structures
        mfcc = load(sprintf('mfcc_%s.mat',task{k}));
        mfccs = mfcc.mfccs;
        
        pitch = load(sprintf('pitch_%s.mat',task{k}));
        f0 = pitch.f0;

        %equalize the data length
        %get the neural data
        resp = double(EEG.data');
        
        %determine the length of the features
        [~,midx] = min([size(mfccs,1) length(f0) length(env) length(resp)]);
        
        if midx == 1
            
            f0 = f0(1:size(mfccs,1));
            env = env(1:size(mfccs,1));
            resp = resp(1:size(mfccs,1),:);
            
        elseif midx == 2
            
            mfccs= mfccs(1:length(f0),:);
            env = env(1:length(f0));
            resp = resp(1:length(f0),:);
            
        elseif midx == 3
            
            mfccs= mfccs(1:length(env),:);
            f0 = f0(1:length(env));
            resp = resp(1:length(env),:);
            
        else

            f0 = f0(1:size(resp,1));
            env = env(1:size(resp,1));
            mfccs = mfccs(1:size(resp,1),:);
        end
               
        %remove the onsets above the cutoff of, needs to be done via loop
        %--> ons of 3s i need the previous one to compute similarity
        ons_idx_sav = [];
        win_len = 20;
        for on = 2:length(ons_idx)
            if ons_dif(on) < max_value && ons_dif(on) >=win_len
                %save the index and that of the preceeding ons
                
                if ons_idx(on)+win_len< length(mfccs) 
                    
                    %cosine similarity
                    %get the mel spectrogram of the corresponding segments
                    melspec_pre = normalize(mfccs(ons_idx(on-1):ons_idx(on-1)+win_len,:),2,'range');
                    melspec_post = normalize(mfccs(ons_idx(on):ons_idx(on)+win_len,:),2,'range');
                    
                    env_pre = normalize(env(ons_idx(on-1):ons_idx(on-1)+win_len,:),'range');
                    env_post = normalize(env(ons_idx(on):ons_idx(on)+win_len,:),'range');
                    
                    
                    pitch_pre = normalize(f0(ons_idx(on-1):ons_idx(on-1)+win_len),'range');
                    pitch_post = normalize(f0(ons_idx(on):ons_idx(on)+win_len),'range');
                    
                    %get the cosine similiarity
                    mel_cos = dot(melspec_post(:),melspec_pre(:)) / (norm(melspec_post(:))*norm(melspec_pre(:)));
                    mel_dis = dtw(melspec_pre,melspec_post)/size(melspec_post,1);
                    
                    %good old MSE
                    env_mse = mean((env_post - env_pre).^2);
                    
                    %get the correlation
                    env_cor = corr(env_post,env_pre);
                    
                    %pitch similarity
                    
                    pitch_dis = dtw(pitch_post,pitch_pre)/length(pitch_pre);
                    pitch_mse = mean((pitch_post - pitch_pre).^2);
                    
                    %                     exp( -sum( (f0(ons_idx(on):ons_idx(on)+win_len) - f0(ons_idx(on-1):ons_idx(on-1)+win_len) ).^2) / (std(f0(ons_idx(on-1):ons_idx(on-1)+win_len)).^2 + std(f0(ons_idx(on):ons_idx(on)+win_len)).^2));
                    %                     pitch_sim = corr(f0(ons_idx(on):ons_idx(on)+win_len),f0(ons_idx(on-1):ons_idx(on-1)+win_len));
                    %envelope similarity
                    
                    %save the index and that of the preceeding ons
                    ons_idx_sav = [ons_idx_sav; ons_idx(on-1) ons_idx(on) ons_dif(on) sum(diff(env_pre(1:5))) sum(diff(env_post(1:5))) max(env(ons_idx(on):ons_idx(on)+win_len,:)) mel_cos mel_dis env_mse env_cor pitch_dis pitch_mse];
                    
                    % save the onset ramp of the post sound
                    
                end
            end
        end
        
%         %normalize the results
%         sim_norm = normalize(ons_idx_sav,'range');
%         
%         %reverse the scale of the distance metrics (small is good)
%         sim_norm(:,[2 3 5 6]) = 1-sim_norm(:,[2 3 5 6]); 
%         
        %remove the lengthy ones
        ons_idx_sav(find(ons_idx_sav(:,2)+50 > EEG.pnts),:) = [];
        %remove onsets that are closer than 1 second -> overlap 
        ons_idx_sav(find(ons_idx_sav(:,3) < min_value),:) = [];
        
        %create the onset vector based on the proceeding onsets
        onset = zeros(length(EEG.data),1);
        onset(ons_idx_sav(:,2),1) = 1;
        
        %and now extract the corresponding neural model ... 
        %maybe epoch?
        EEG_epo = OT_epochize(EEG,onset,ep_t,0);
        erp_time = linspace(ep_t(1),ep_t(2),size(EEG_epo,2));
        
        EEG.data = EEG_epo;
        EEG.times = erp_time*1000;
        EEG.trials = size(EEG_epo,3);
        EEG.pnts = size(EEG_epo,2);
        %% Baseline Correction 
        base_idx = [ep_t(1)+0.01 -0.01]*EEG.srate;
        b = dsearchn(erp_time',base_idx');
        
        
        %subtract the data 
        EEG_rm = pop_rmbase(EEG,[-100 -10]);
        
        %save the erp structure
        sav_eeg{s,k} = EEG_rm.data;
        
        %save the euclidean distance
        sav_dist{s,k} = ons_idx_sav;
        
        fprintf('participant %s \r',sbj{s})
    
    end  
end

%% maybe save the data you fool 
cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\audiObj\')
DNS_SingleOnsets = struct();
DNS_SingleOnsets.sav_eeg = sav_eeg;
DNS_SingleOnsets.sav_dist = sav_dist;
DNS_SingleOnsets.t = 'extended aud obj analysis but with everything';


save('DNS_ISI_audiobj_extended_all','-struct','DNS_SingleOnsets')




        
fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\auditory_Obj\ '


temp_dist = [];
%get the global distribution 
for s = 1:length(sbj)
   
    temp_ecld = cat(1,sav_dist{s,:});
    
    temp_dif{s,:} = temp_ecld(:,3);
    
    temp_envdif{s,:} = temp_ecld(:,[4 5]);
    
    temp_envmax{s,:} = temp_ecld(:,6);
    
    %cut the first 3 rows
    temp_ecld(:,[1 2 3 4 5 6])= [];
    
    %normalize
    ecld_norm = normalize(temp_ecld,'range');
    
    %invert the distance measures -> MSE and so forth 
    ecld_norm(:,[2 3 5 6]) = 1-ecld_norm(:,[2 3 5 6]);
    
    %compute the weighted composite score
    ecld_score = sum(ecld_norm.*0.1667,2);
    
    temp_dist = [temp_dist; ecld_score];
    
    com_score{s,:} = ecld_score;
    
end

figure
hist(temp_dist)

%get the global bin edges
bin2 = quantile(temp_dist, linspace(0, 1, 3 + 1));
% bin2 = binEdges{1,2};

%plot the results and sort them according to their euclidean distance
agg_dat=[];
temp_eeg = [];

 for s = 1:length(sbj)
    
    temp_dat = cat(3,sav_eeg{s,:});
    temp_ecld = com_score{s,:};
    
    temp_eeg = cat(3,temp_eeg,temp_dat);
    
    dif_scores = temp_dif{s,:};
    env_dif = temp_envdif{s,:};
    max_env = temp_envmax{s,:};
%     
    %sort the data according to ecld
    [temp_ecld_sort ecld_idx] = sort(temp_ecld,'descend');
    temp_dat = squeeze(mean(temp_dat(:,:,ecld_idx),1));
    dif_scores = dif_scores(ecld_idx,1);
    env_dif = env_dif(ecld_idx,:);
    max_env = max_env(ecld_idx,:);

% %sort the data according to ecld
%     [dif_scores ecld_idx] = sort(dif_scores,'descend');
%     temp_dat = squeeze(mean(temp_dat(:,:,ecld_idx),1));
%     temp_ecld = temp_ecld(ecld_idx,1);
%     env_dif = env_dif(ecld_idx,:);
%     max_env = max_env(ecld_idx,:);
    
    num_trials(s,:) = size(temp_ecld); 
       
    binIndices = discretize(temp_ecld_sort, bin2);
    %average over these fuckers
    for bi = 1:length(unique(binIndices))
        agg_dat(s,:,bi) = mean(temp_dat(:,binIndices == bi),2);
        dif_score(s,bi) =  mean(dif_scores(binIndices == bi));
        env_score(s,bi) = mean(env_dif(binIndices == bi,2));
        env_maxscore(s,bi) = mean(max_env(binIndices == bi));
        sim_score(s,bi) = mean(temp_ecld(binIndices == bi));
    end


%     %only take the bottom and top 20 trials
%     agg_dat(s,:,1) = mean(temp_dat(:,1:40),2);
%     agg_dat(s,:,2) = mean(temp_dat(:,end-41:end),2);
    
    
    
end

%% linear mixed modeling
for i = 1:length(sbj)
    figure
    plot(squeeze(mean(agg_dat(i,:,:),3)))
end
N1 = [0.04 0.11];
P2 = [0.180 0.25];

erp_time = ep_t(1):1/100:ep_t(2);

n1_idx = dsearchn(erp_time',N1');
p2_idx = dsearchn(erp_time',P2');

participant_ids = repelem((1:20)', num_trials(:,1));

%category labels
categories = {'VeryDifferent', 'Mediocre', 'MostSimilar'};

% Create a repeating cycle where each subject has the three categories
categoryLabels = repmat(categories, 1, length(sbj))'; % Transpose to make column vector
categoryLabels = categorical(categoryLabels); % Convert to categorical

% single trial data
neural_data = double(squeeze(mean(temp_eeg(:,n1_idx(1):n1_idx(2),:),[1 2])));
similarity_scores = cat(1,cell2mat(com_score));
sharpness_values  = cat(1,cell2mat(temp_envdif));
intensity_values  = cat(1,cell2mat(temp_envmax));
distance_values  = cat(1,cell2mat(temp_dif));

tbl = table(participant_ids, neural_data, similarity_scores, sharpness_values(:,2), intensity_values, distance_values, ...
    'VariableNames', {'Participant', 'NeuralData','SimilarityScore', 'Sharpness', 'Intensity', 'Distance'});

[correlation_matrix, pval] = corr([tbl.Sharpness, tbl.Intensity, tbl.Distance tbl.SimilarityScore]);

% Display correlation matrix
disp('Correlation Matrix:');
disp(correlation_matrix);

lme_all = fitlme(tbl, ...
    'Distance  ~  Sharpness * Intensity + (1|Participant)');

disp(lme_all)

participant_ids = repelem((1:20)', 3);

%averaged data
neural_data = reshape(squeeze( max(agg_dat(:,p2_idx(1):p2_idx(2),:),[],2) - min(agg_dat(:,n1_idx(1):n1_idx(2),:),[],2))',[],1);
similarity_scores = normalize(reshape(sim_score',[],1),'zscore');
sharpness_values = normalize(reshape(env_score',[],1),'zscore');
intensity_values = normalize(reshape(env_maxscore',[],1),'zscore');
distance_values = normalize(reshape(dif_score',[],1),'zscore');

dataTable = table(participant_ids,  neural_data, distance_values, sharpness_values, intensity_values, similarity_scores,...
    'VariableNames', {'Subject', 'NeuralResponse', 'Distance', 'Sharpness', 'Intensity','SimilarityScore'});

% Display first few rows to verify structure
% disp(dataTable);

[correlation_matrix, pval] = corr([dataTable.Sharpness, dataTable.Intensity, dataTable.Distance, dataTable.SimilarityScore]);

% Display correlation matrix
disp('Correlation Matrix:');
disp(correlation_matrix);


lme_all = fitlme(dataTable, ...
    'NeuralResponse ~ Distance * Sharpness * Intensity * SimilarityScore + (1|Subject)');

disp(lme_all)

lme_dist = fitlme(dataTable, ...
    'NeuralResponse ~ Distance + (1|Subject)');
% Display results
disp(lme_dist);

lme_sharp = fitlme(dataTable, ...
    'NeuralResponse ~ Sharpness + (1|Subject)');
% Display results
disp(lme_sharp);

lme_sim = fitlme(dataTable, ...
    'NeuralResponse ~ SimilarityScore + (1|Subject)');
% Display results
disp(lme_sim);

lme_int = fitlme(dataTable, ...
    'NeuralResponse ~ Intensity + (1|Subject)');
% Display results
disp(lme_int);

% Display results
disp(lme);

lme_interaction = fitlme(dataTable, ...
    'NeuralResponse ~ SimilarityScore * Category + Sharpness * Category + Intensity * Category + (1|Subject)');

disp(lme_interaction);


%% plot the distance between tones 
%i want to know whether there is a systematic difference between sound
%events due to one group being closer together
g = {'very different','mediocre','most similar'};
figure
boxplot(dif_score,'Labels',g)
title('Distance of sound events')

%test the significance
[h,p,ci,stats]  = ttest(dif_score)

%% envelope differences
figure
boxplot(env_score,g)
title('Sharpness of Onsets')

[p, tbl, stats] = anova1(env_score,g)
results = multcompare(stats);

%% envelope max
figure
boxplot(env_maxscore,g)
title('Intensity of Sound Events')

%% determine whether there is an onset difference



%% plot the ERPs
figure, hold on

fill([0.04 0.04 0.11 0.11], [-1.5 2 2 -1.5], [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

for i = 1:size(agg_dat,3)
%     if i == 2; continue; end
    plot_dat = agg_dat(:,:,i);
    plot_dat(12,:) = [];
    %Compute the mean and SEM
    meanData = mean(plot_dat,1); % Mean across rows (observations)
    SEM = std(plot_dat, 0, 1) ./ sqrt(size(plot_dat, 1)); % SEM calculation
    
    % Time vector (assume 1 unit per time point)
%     time = linspace(-100, 500, size(plot_dat, 2)); % Adjust as per your time points
    
    % Plot the mean
    p(i) = plot(erp_time, meanData, 'LineWidth', 2); % Mean line (red)
    c_lor(i,:) = get(p(i),'Color');
    % Plot shaded SEM as error bands
    fill([erp_time, fliplr(erp_time)], ...
        [meanData + SEM, fliplr(meanData - SEM)], ...
        c_lor(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'k'); % Shaded area
%     figure
%     imagesc(plot_dat)
    %
end
legend(p,{'very different','mediocre','most similar'},'box','off','FontSize',14)



% legend(p,{'most similar','moderate','weakly','very different'},'box','off')
xlabel('Time in m.s.')
ylabel('microvolts')
set(gca,'FontSize',16)

% save_fig(gcf,fig_path,'DNS_ISI_audi_obj')


%% test whether there are potential P2 differences

N1 = [0.04 0.11];
P2 = [0.180 0.25];

n1_idx = dsearchn(erp_time',N1');
p2_idx = dsearchn(erp_time',P2');

%contrast N1
n1_dat = squeeze(min(agg_dat(:,n1_idx(1):n1_idx(2),:),[],2));
[n1_val,~,n1_stat] = signrank(n1_dat(:,1),n1_dat(:,3));

%contrast P2
p2_dat = max(agg_dat(:,p2_idx(1):p2_idx(2),:),[],2);
[p2_val,~,p2_stat] = signrank(p2_dat(:,2),p2_dat(:,3));

%contrast P2-N1 complex
dif_dat = squeeze(max(agg_dat(:,p2_idx(1):p2_idx(2),:),[],2)) - squeeze(min(agg_dat(:,n1_idx(1):n1_idx(2),:),[],2));
[dif_val,~,dif_stat] = signrank(dif_dat(:,2),dif_dat(:,1));


figure
violinplot(dif_dat,{'most similar','moderate','weakly','very different'},...
    'ViolinColor',c_lor);
        

%convert the data to fieldtrip 
ft_dat = cell(2,length(sbj)); %1= low; 2=high
for s=1:length(sbj)

    EEG = [];
    [EEG,PATH] = OT_preprocessing(s,k,sbj,40);
    EEG.pnts = length(erp_time);
    EEG.times = erp_time/1000;
    EEG.xmin = min(erp_time);
    EEG.xmax = max(erp_time);
    EEGl = EEG;
    EEGh = EEG;
    
    
    EEGl.data = squeeze(agg_dat(s,:,1))
    EEGh.data = squeeze(agg_dat(s,:,3))
    
    ft_dat{1,s} = eeglab2fieldtrip(EEGl,'timelock');
    ft_dat{2,s} = eeglab2fieldtrip(EEGh,'timelock');
    chan_layout= eeglab2fieldtrip(EEGh,'chanloc');

end
ft_l = ft_dat(1,:);
ft_h = ft_dat(2,:);

cfg_neighb        = [];
cfg_neighb.method = 'distance';
cfg_neighb.neighbourdist = 80;
neighbours = ft_prepare_neighbours(cfg_neighb, ft_l{1,1});


cfg         = [];
cfg.channel = {'EEG'};
cfg.latency = [0 0.5];
cfg.method           = 'montecarlo';
cfg.statistic        = 'depsamplesT';
cfg.correctm         = 'cluster';
cfg.clusteralpha     = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan        = 2;
cfg.neighbours       = neighbours;  % same as defined for the between-trials experiment
cfg.tail             = 0;
cfg.clustertail      = 0;
cfg.alpha            = 0.05;
cfg.numrandomization = 1000;


Nsubj  = length(sbj);

design = zeros(2, Nsubj*2);
design(1,:) = [1:Nsubj 1:Nsubj];
design(2,:) = [ones(1,Nsubj) ones(1,Nsubj)*2];

cfg.design = design;
cfg.uvar   = 1;
cfg.ivar   = 2;
[stat] = ft_timelockstatistics(cfg, ft_l{:}, ft_h{:});

%% get the data to plot ready
cfg = [];
cfg.channel   = 'all';
cfg.latency   = [0 0.5];
cfg.parameter = 'avg';
GA_l        = ft_timelockgrandaverage(cfg, ft_l{:});
GA_h         = ft_timelockgrandaverage(cfg, ft_h{:});

%         figure
cfg = []
cfg.layout = chan_layout;
cfg.parameter = 'avg'
ft_multiplotER(cfg,GA_h,GA_l)
%save_fig(gcf,fig_path,sprintf('DNS40_multi_%s_%s',auditory{ao}, task{k}))


cfg = [];
cfg.operation = 'subtract';
cfg.parameter = 'avg';
GA_lvsh    = ft_math(cfg, GA_l, GA_h);

%% bojana code for plotting

comps={'aMINUSi'};
Cluster=[];
for idxComps = 1:size(comps,2)
    CustMap=eval(['Map_' comps{idxComps}]);
    
    if isfield(stat,'posclusters') && isempty(stat.posclusters) ==0
        pos_cluster_pvals = [stat.posclusters(:).prob];
        pos_signif_clust = find(pos_cluster_pvals < stat.cfg.clusteralpha);%stat.cfg.clusteralpha
        
        if ~isempty(pos_signif_clust)
            for idxPos = 1:length(pos_signif_clust)
                pos = stat.posclusterslabelmat == pos_signif_clust(idxPos);
                sigposCLM = (pos == 1);
                probpos(idxPos) = stat.posclusters(idxPos).prob;
                possum_perclus = sum(sigposCLM,1); %sum over chans for each time- or freq-point
                
                Cluster{idxComps,idxPos}=sum(pos,2)>0;
                
                ind_min = min(find(possum_perclus~=0));
                ind_max = max(find(possum_perclus~=0));
                time_perclus = [stat.time(ind_min) stat.time(ind_max)];%
                ClusterTime{idxComps,idxPos}=time_perclus;
                
                % provides the OG chanlocs indicies
                pos_int = find(sum(pos(:,ind_min:ind_max),2)> 1);
                pos_chan = eeg_chan(pos_int);
                
                
                figure('Name',['Probability:' num2str(probpos(idxPos))],'NumberTitle','off')
                cfgPlot=[];
                cfgPlot.xlim=[time_perclus(1) time_perclus(2)];
                cfgPlot.highlight = 'labels';
                cfgPlot.highlightchannel = pos_chan;
                cfgPlot.highlightsymbol    = 'x'
                cfgPlot.highlightcolor     = [220 20 60]./255
                cfgPlot.highlightsize      = 8
                cfgPlot.parameter='avg';
                cfgPlot.layout= layout_m;
                %                         cfgPlot.colormap = CustMap;
                %                         cfgPlot.zlim = [0 4];
                cfgPlot.colorbar ='EastOutside';
                cfgPlot.comment = sprintf('%s [%.3f %.3f]s.',auditory{ao},cfgPlot.xlim(1),cfgPlot.xlim(2))
                cfgPlot.commentpos = 'title';
                cfgPlot.colorbar ='yes';
                cfgPlot.style = 'straight';
                stat.stat = abs(stat.stat);
                ft_topoplotER(cfgPlot, GA_lvsh);
                %                         save_fig(gcf,[fig_path '\OT14eventdns\TRF\'],sprintf('TRF_posc_%s_%.3f_%.3f',auditory{ao},cfgPlot.xlim(1),cfgPlot.xlim(2)))
                %                         save_fig(gcf,[fig_path '\OT14eventdns\ERP\'],sprintf('DNS40epo_ovlp_posc_%s_%.3f_%.3f',auditory{ao},cfgPlot.xlim(1),cfgPlot.xlim(2)))
                %                         save_fig(gcf,fig_path,sprintf('TRF_%s_pos',auditory{ao}))
            end
        end
    end
    
    if isfield(stat,'negclusters') && isempty(stat.negclusters) ==0
        neg_cluster_pvals = [stat.negclusters(:).prob];
        neg_signif_clust = find(neg_cluster_pvals < stat.cfg.clusteralpha);%stat.cfg.clusteralpha
        
        if isempty(neg_signif_clust) == 0
            for idxNeg = 1:length(neg_signif_clust)
                neg = stat.negclusterslabelmat == neg_signif_clust(idxNeg);
                signegCLM = (neg == 1);
                probneg(idxNeg) = stat.negclusters(idxNeg).prob;
                negsum_perclus = sum(signegCLM,1); %sum over chans for each time- or freq-point
                
                Cluster{idxComps,length(neg_signif_clust)+idxNeg}=sum(neg,2)>0;
                
                ind_min = min(find(negsum_perclus~=0));
                ind_max = max(find(negsum_perclus~=0));
                time_perclus = [stat.time(ind_min) stat.time(ind_max)];%
                ClusterTime{idxComps,length(neg_signif_clust)+idxNeg}=time_perclus;
                
                % provides the OG chanlocs indicies
                neg_int = find(sum(neg(:,ind_min:ind_max),2)> 1);
                neg_chan = eeg_chan(neg_int);
                
                figure('Name',['Probability:' num2str(probneg(idxNeg))],'NumberTitle','off')
                cfgPlot=[];
                cfgPlot.xlim=[time_perclus(1) time_perclus(2)];
                cfgPlot.highlight = 'labels';
                cfgPlot.highlightchannel = neg_chan;
                cfgPlot.highlightsymbol    = 'x'
                cfgPlot.highlightcolor     = [220 20 60]./255
                cfgPlot.highlightsize      = 8
                cfgPlot.parameter='avg';
                cfgPlot.layout=layout_m;
                %                         cfgPlot.colormap = CustMap;
                %                         cfgPlot.zlim = [0 4];
                cfgPlot.comment = sprintf('%s [%.3f %.3f]s.',auditory{ao},cfgPlot.xlim(1),cfgPlot.xlim(2))
                cfgPlot.colorbar ='EastOutside';
                cfgPlot.commentpos = 'title';
                cfgPlot.colorbar ='yes';
                cfgPlot.style = 'straight';
                stat.stat = abs(stat.stat);
                ft_topoplotER(cfgPlot, GA_lvsh);
                %                         save_fig(gcf,[fig_path '\OT14eventdns\TRF\'],sprintf('TRF_negc_%s_%.3f_%.3f',auditory{ao},cfgPlot.xlim(1),cfgPlot.xlim(2)))
                
                %                         save_fig(gcf,[fig_path '\OT14eventdns\ERP\'],sprintf('DNS40epo_ovlp_negc_%s_%.3f_%.3f',auditory{ao},cfgPlot.xlim(1),cfgPlot.xlim(2)))
                %                         save_fig(gcf,fig_path,sprintf('TRF_%s_neg',auditory{ao}))
                
            end
        end
    end
 