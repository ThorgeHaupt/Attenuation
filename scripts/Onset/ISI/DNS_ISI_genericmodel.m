%% lets take all the participants and compute the model over them ... 
OT_setup 
DNS_setup


%partition the data set
nfold = 12;
testfold = 1;

%number of bins
min_value = 1;       % Minimum value of the range
max_value = 1000;    % Maximum value of the range

% uniform distribution
numBins = linspace(2,8,7); % Number of bins

%load the distance collection vector
load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\dns_dist_descriptives.mat')

%clean it first
dns_dist(dns_dist > max_value) = [];


% uniform distribution
for i = 1:length(numBins)
    binEdges{i} = quantile(dns_dist, linspace(0, 1, numBins(i) + 1));
end


for s=1:length(sbj)
      
    
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab('nogui');

    for k=1:2
        
        %% compute the envelopes
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        
        cd(PATH)
        
        
        novelty_ultm = load(sprintf('ons_ult_%s',task{k}));
        
        fs_new = EEG.srate;
        
        peak = smooth_peak(novelty_ultm.novelty_ultm,fs_new,'sigma',4);
        
        stim = peak';
        
        %get the neural data
        resp = double(EEG.data');
        
        if size(resp,1)>size(stim,1)
            resp = resp(1:size(stim,1),:);
        elseif size(resp,1)<size(stim,1)
            stim = stim(1:length(resp),:);
        end
        EEG.data = resp';
        eeg_sav{k} = EEG; 
        peak_sav{k} = stim;
        
    end
    
    EEG = pop_mergeset(eeg_sav{1},eeg_sav{2});
    
    peak = [peak_sav{1}' peak_sav{2}']';
    
    %get the indicies
    ons_idx = find(peak);
    
    %find the distance
    ons_dif = diff(ons_idx);
    
    %remove the first one
    ons_idx(1) = [];
    
    %remove the onsets above the cutoff of
    ons_del = find(ons_dif > max_value);
    ons_idx(ons_del) = [];
    ons_dif(ons_del) = [];

    for ed = 1:length(numBins)
        
        
        [counts, bins, binIndices] = histcounts(ons_dif,binEdges{ed});
        
        %create the feature vector based on the random entries
        ons_bin = zeros(length(counts),length(peak));
        for i = 1:length(binIndices)
            ons_bin(binIndices(i),ons_idx(i)) = 1;
        end
        
        stim_cell{s,ed} = ons_bin';
    end
    
    if ed == 7
        stim_cell{s,ed+1} = sum(stim_cell(s,1),2);
    end
    %save the corresponding EEG data
    EEG_cell{s} = EEG.data'
    
end


for ed = 5:length(numBins)+1
    temp_w = [];
    temp_r = [];
    for tr = 1:length(sbj)
        
        idx = true(size(sbj));
        idx(tr) = false;
        stest = stim_cell{tr,ed};
        rtest = EEG_cell{tr};
        
        strain =  stim_cell(idx,ed);
        rtrain = EEG_cell(idx);

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
        
        
        %save the weights
        temp_w(:,:,:,tr) = model_train.w;
        
        %predict the neural data
        [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
        
        %save the prediction values
        
        fprintf('DIST GRID: Bin = %d, sbj: %d \r',ed,tr)
        result_reg(ed,tr,:) =  STATS.r;
        
    end
    mlpt_weight{ed} = squeeze(mean(temp_w,4));
    
    
end


%% save the results
trf_time = model_train.t;


cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\')

DNS_uni = struct();
DNS_uni.result_reg = result_reg;
DNS_uni.mlpt_weight = mlpt_weight;
DNS_uni.numBins= numBins;
DNS_uni.trf_time = trf_time;
DNS_uni.binEdges = binEdges;
% DNS_grid.auditory = auditory;

DNS_uni.t = 'Train and test on a generic model';

save('DNS_ISI_generic.mat','-struct','DNS_uni')


%% plot the results
fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\Uniform\generic\';

fig_pos = [121,222,504,437];

res = mean(result_reg,3)
%sort the rest
[reso_sort s_idx] = sort(res(end,:),'ascend')
reso_sort = res(:,s_idx);

%compare these bitches statistically 
figure, hold on
set(gcf,'Position',fig_pos);
for ed = 1:length(numBins)
    
    p_val(ed) = signrank(reso_sort(ed,:),reso_sort(end,:));
    plot(reso_sort(ed,:),'o-','Color', gradientColors_ISI(ed,:),'linew',2)
end
plot(reso_sort(end,:),'o-','Color',[0, 0.5, 0.7410],'linew',2)
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_val,0.05,'dep');

xlabel('Subjects');
ylabel('Pearson''s r');
legend([num2str(numBins'); num2str(1)],'Location', 'NorthWest','Box','off');
grid off;
set(gca,'FontSize',14,'Ylim',[0.01 0.08])

% save_fig(gcf,'fig_path','DNS_ISI_generic_pred')

figure
violinplot(squeeze(mean(result_reg,3))')
sigstar({[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8]},adj_p)


%% plot the model weights

for nb = 1:length(numBins)+1
    
    edge = binEdges{nb};
    
    %% get the corresponding weights
    %get the y-values
    ylabels = round(edge(2:end)./100,2);
    figure
%     set(gcf,'position',fig_pos)
    %plot the weights
    t = tiledlayout(1,1)
    
    weight_temp = mlpt_weight{:,nb};
    
    plot_dat = squeeze(mean(weight_temp,3))';
    nexttile
    for i =1:size(plot_dat,2)
        plot(trf_time,plot_dat(:,i)+(i*14),'Color',[0.49, 0.18, 0.56],'linew',2)
        y_val(i) = mean(plot_dat(1:10,i)+(i*14));
        hold on
    end
    grid off
    title('Model Weights')
    xlabel('Time Lag (ms)');
    set(gca,'FontSize',16,'YTick',round(y_val),'YtickLabel',ylabels)
    box off
end



